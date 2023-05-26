"""
This file trains the policy network using the U-Net architecture
as a policy evaluation step.
How to Run on Different Benchmarks:
    python pretrain.py --model R32_C10, R32_C100, R34_fMoW, R50_ImgNet
       --lr 1e-4
       --cv_dir checkpoint directory
       --batch_size 1048 (Higher is better)
       --ckpt_hr_cl Load the checkpoint from the directory for HR classifier
       --lr_size 8, 56 (Depends on the dataset)
"""
import os
from tensorboard_logger import configure, log_value
import torch
from torch.autograd import Variable
# !!! The Variable API has been deprecated: Variables are no longer necessary to use autograd with tensors.
# Autograd automatically supports Tensors with requires_grad set to True !!!
import torch.utils.data as torchdata
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import torch.optim as optim

from torch.distributions import Multinomial, Bernoulli
from utils import utils
from Seg_models_pytorch import get_model_pytorch
from Seg_models import get_model_keras

import torch.backends.cudnn as cudnn

cudnn.benchmark = True

IMAGE_SIZE = (256, 256)
N_CHANNELS = 3
N_FILTERS = 16
WEIGHTS_FILE ='cv/tmp/Landsat-8/unet/model_unet_voting_final_weights.h5'

import argparse
parser = argparse.ArgumentParser(description='Policy Network Finetuning-I')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--model', default='R_Landsat8', help='R<depth>_<dataset> see utils.py for a list of configurations')
parser.add_argument('--ckpt_hr_cl', help='checkpoint directory for the high resolution classifier')
parser.add_argument('--data_dir', default='data/', help='data directory')
parser.add_argument('--load', default=None, help='checkpoint to load agent from')
parser.add_argument('--cv_dir', default='cv/tmp/', help='checkpoint directory (models and logs are saved here)')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--max_epochs', type=int, default=20, help='total epochs to run')
parser.add_argument('--parallel', action ='store_true', default=False, help='use multiple GPUs for training')
parser.add_argument('--penalty', type=float, default=-0.5, help='to penalize the PN for incorrect predictions')
parser.add_argument('--alpha', type=float, default=0.8, help='probability bounding factor')
parser.add_argument('--lr_size', type=int, default=32, help='Policy Network Image Size')
parser.add_argument('--test_interval', type=int, default=5, help='At what epoch to test the model')
args = parser.parse_args()

if not os.path.exists(args.cv_dir):
    os.system('mkdir ' + args.cv_dir)
utils.save_args(__file__, args)

IMG_PATH = 'data/images'
MSK_PATH = 'data/voting_masks'
TH_FIRE = 0.25 # fire threshold

def visualize(img3c, mask):
    # permute for visualization purposes
    img3c = img3c.float().permute(1, 2, 0)
    mask = mask.float().permute(1, 2, 0)

    plt.subplot(1, 2, 1)
    plt.imshow(img3c)
    plt.title('Original image 3c')

    plt.subplot(1, 2, 2)
    plt.imshow(mask)
    plt.title('Voting mask (target)')

    plt.show()
def get_SegNet_prediction(images):
    keras_unet = get_model_keras(model_name='unet',
                                 input_height=IMAGE_SIZE[0], input_width=IMAGE_SIZE[1],
                                 n_filters=N_FILTERS, n_channels=N_CHANNELS)
    keras_unet.load_weights(WEIGHTS_FILE)
    images = images.permute(0, 2, 3, 1)
    y_pred = keras_unet.predict(np.array(images.cpu()), batch_size=args.batch_size)

    y_pred = y_pred[:, :, :, 0] > TH_FIRE
    pred_masks = np.array(y_pred * 255, dtype=np.uint8)
    return pred_masks

def dice_coefficient(predicted, target):
    smooth = 1.  # Smoothing factor to avoid division by zero
    intersection = torch.sum(predicted * target)
    union = torch.sum(predicted) + torch.sum(target)
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

def compute_SegNet_reward(preds, targets, policy):
    """
    :param preds: the predicted segmentation masks
    :param targets: the ground truth segmentation masks
    :param policy: binary vector indicating sampled or non-sampled patches
    """
    preds[preds==255] = 1

    patch_use = policy.sum(1).float() / policy.size(1) # I will penalize the policy with the number of sampled patches
    sparse_reward = 1.0 - patch_use**2

    # dice = dice_coef(targets, preds)
    dice = dice_coefficient(preds, targets)
    penalty = 1-dice # the segmentation error between 0 and 1

    reward = sparse_reward - penalty

    return reward.unsqueeze(1)

def train(epoch):

    agent.train() # trains the policy network only

    rewards, rewards_baseline, policies = [], [], []

    for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(trainloader), total=len(trainloader), disable=True):

        inputs = inputs.float().permute(0, 3, 1, 2)
        targets = targets.float().permute(0, 3, 1, 2)
        # if args.parallel:
        # inputs = inputs.cuda()
        # targets = targets.cuda()

        inputs_agent = inputs.clone()
        inputs_map = inputs.clone()
        inputs_sample = inputs.clone()

        # Run the low-res image through Policy Network
        inputs_agent = torch.nn.functional.interpolate(inputs_agent, (args.lr_size, args.lr_size))

        outputs_agent = agent.forward(inputs_agent, 'R_Landsat8', 'lr')
        probs = torch.sigmoid(outputs_agent)
        probs = probs*args.alpha + (1-args.alpha) * (1-probs) # temperature scaling (to encourage exploration)

        # --Got one prob vector for each image of the batch--

        distr = Bernoulli(probs) # batch_size x patch_size

        policy_sample = distr.sample()

        # Test time policy - used as baseline policy in the training step
        policy_map = probs.data.clone()
        policy_map[policy_map<0.5] = 0.0
        policy_map[policy_map>=0.5] = 1.0

        # Agent sampled high resolution images
        inputs_map = utils.agent_chosen_input(inputs_map, policy_map, mappings, patch_size)

        inputs_sample = utils.agent_chosen_input(inputs_sample, policy_sample.int(), mappings, patch_size)
        # agent_chosen_input: get patches of input based on policy arg 2

        # --POLICY NETWORK DONE--

        # for i in range(5):
        #     plt.imshow( inputs_sample[i].cpu().permute(1, 2, 0) )
        #     plt.show()

        # Forward propagate (masked HR) images through the Segmentation Network

        # visualize(inputs_map[0].cpu(), targets[0].cpu())
        preds_map = get_SegNet_prediction(inputs_map)
        preds_sample = get_SegNet_prediction(inputs_sample)

        # Find the reward for baseline and sampled policy
        preds_map = torch.from_numpy(preds_map[:, None, :, :])
        preds_sample = torch.from_numpy(preds_sample[:, None, :, :])
        reward_map = compute_SegNet_reward(preds_map, targets, policy_map.data)
        reward_sample = compute_SegNet_reward(preds_sample, targets, policy_sample.data)
        # advantage = reward_sample.cuda().float() - reward_map.cuda().float()
        advantage = reward_sample.float() - reward_map.float()
        # print("adv:", advantage) # batch_size x 1 -> 1 adv. for each image

        # Find the loss for only the policy network
        loss = -distr.log_prob(policy_sample)  # formula (9)
        loss = loss * Variable(advantage).expand_as(policy_sample) # REINFORCE
        # print(loss.size()) # batch_size x patch_size -> 1 loss for each patch of each image
        loss = loss.mean() # negative value for the loss is normal in policy gradient

        optimizer.zero_grad()
        loss.backward() # compute gradients of loss w.r.t. weights of policy network
        optimizer.step() # run Adam optimizer to update the weights

        rewards.append(reward_sample.cpu())
        rewards_baseline.append(reward_map.cpu())
        policies.append(policy_sample.data.cpu())

    reward, sparsity, variance, policy_set = utils.performance_stats(policies, rewards)
    # torch.cuda.empty_cache()
    print('Train: %d | Rw: %.3f | S: %.3f | V: %.3f | samples: %d'%(epoch, reward, sparsity, variance, len(policy_set)))
    # log_value('train_accuracy', accuracy, epoch)
    # log_value('train_reward', reward, epoch)
    # log_value('train_sparsity', sparsity, epoch)
    # log_value('train_variance', variance, epoch)
    # log_value('train_baseline_reward', torch.cat(rewards_baseline, 0).mean(), epoch)
    # log_value('train_unique_policies', len(policy_set), epoch)

def test(epoch):

    agent.eval() # flag: deactivate training (gradient update) mode

    rewards, policies = [], []
    for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(testloader), total=len(testloader)):
        # UserWarning: volatile was removed and now has no effect. Use `with torch.no_grad():` instead
        with torch.no_grad():
            # if (args.model == 'R_Landsat8'):
            inputs = inputs.float().permute(0,3,1,2)
            # targets = torch.Tensor([int(val) for val in targets])

        if args.parallel:
            inputs = inputs.cuda()
            targets = targets.cuda()

        # Get the low resolution agent images
        inputs_agent = inputs.clone()
        inputs_agent = torch.nn.functional.interpolate(inputs_agent, (args.lr_size, args.lr_size))
        probs = torch.sigmoid(agent.forward(inputs_agent, 'R_Landsat8', 'lr'))

        # Sample the test-time policy
        policy = probs.data.clone()
        policy[policy<0.5] = 0.0
        policy[policy>=0.5] = 1.0

        # Get the masked high-res image and perform inference
        inputs = utils.agent_chosen_input(inputs, policy, mappings, patch_size)
        preds = get_SegNet_prediction(inputs)

        preds = torch.from_numpy(preds[:, None, :, :])
        reward = compute_SegNet_reward(preds, targets, policy.data)

        rewards.append(reward)
        policies.append(policy.data)

    reward, sparsity, variance, policy_set = utils.performance_stats(policies, rewards)
    torch.cuda.empty_cache()

    print('Test - Rw: %.3f | S: %.3f | V: %.3f | samples: %d'%(reward, sparsity, variance, len(policy_set)))
    # log_value('test_accuracy', accuracy, epoch)
    # log_value('test_reward', reward, epoch)
    # log_value('test_sparsity', sparsity, epoch)
    # log_value('test_variance', variance, epoch)
    # log_value('test_unique_policies', len(policy_set), epoch)

    # Save the Policy Network - Classifier is fixed in this phase
    # agent_state_dict = agent.module.state_dict() if args.parallel else agent.state_dict()
    # state = {
    #   'agent': agent_state_dict,
    #   'epoch': epoch,
    #   'reward': reward
    #   # 'acc': accuracy
    # }
    # torch.save(state, args.cv_dir+'Landsat-8/Policy_ckpt_E_%d_test_acc_%.3f_R_%.2E'%(epoch, accuracy, reward))

#--------------------------------------------------------------------------------------------------------#
trainset, testset = utils.get_dataset(args.model, args.data_dir)
trainloader = torchdata.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
testloader = torchdata.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0)

# 1. Load the Agent
_, _, agent = utils.get_model(args.model)

# Save the args to the checkpoint directory
# configure(args.cv_dir+'/log', flush_secs=5)

# Agent Action Space
mappings, _, patch_size = utils.action_space_model('Landsat-8')

# 2. Load the segmentation network (Unet-Light-3c) --> den einai to light 3c (malakes)
pytorch_unet = get_model_pytorch(model_name='unet',
                 input_height=IMAGE_SIZE[0], input_width=IMAGE_SIZE[1],
                 n_filters=N_FILTERS, n_channels=N_CHANNELS)
# model.summary()

# keras_unet = get_model_keras(model_name='unet',
#                  input_height=IMAGE_SIZE[0], input_width=IMAGE_SIZE[1],
#                  n_filters=N_FILTERS, n_channels=N_CHANNELS)
#
# print('U-Net loaded!')
#
#
# # 3. Load the weights (trained on voting scheme)
# keras_unet.load_weights(WEIGHTS_FILE)

# img_path = os.path.join('data/images', 'LC08_L1TP_118029_20200816_20200816_01_RT_p00707.tif')
# from Seg_generator import get_img_762bands
# img3c = get_img_762bands(img_path) # in 3 channels
#
# y_pred = keras_unet.predict(np.array([img3c]), batch_size=1)


pytorch_unet.load_state_dict(torch.load('pytorch_unet.pt'))
# unet.eval()
print('Weights loaded!')
# Check if the weights are the same !!!!!

# utils.keras2pytorch_model(keras_unet, pytorch_unet)

# Load the Policy Network (if checkpoint exists)
start_epoch = 1
# if args.load is not None:
#     checkpoint = torch.load(args.load)
#     agent.load_state_dict(checkpoint['agent'])
#     start_epoch = checkpoint['epoch'] + 1
#     print('loaded pretrained agent from', args.load)

# Parallelize the models if multiple GPUs available - Important for Large Batch Size
# if args.parallel:
#     agent = nn.DataParallel(agent)
#     unet = nn.DataParallel(unet)

# unet.eval().cuda()
# agent.cuda() # Only agent is trained

optimizer = optim.Adam(agent.parameters(), lr=args.lr)

for epoch in range(start_epoch, start_epoch+args.max_epochs):
    train(epoch)
    if epoch % args.test_interval == 0:
        test(epoch)
