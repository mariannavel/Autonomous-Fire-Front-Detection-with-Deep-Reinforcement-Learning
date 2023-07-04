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
# Autograd automatically supports Tensors with requires_grad set to True !!!
import torch.utils.data as torchdata
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import torch.optim as optim

from torch.distributions import Multinomial, Bernoulli
from utils import utils
from Seg_models_pytorch import get_model_pytorch
# from Seg_models import get_model_keras
from visualize import visualize_with_seg_mask, visualize_images
# from torchsummary import summary
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
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--max_epochs', type=int, default=100, help='total epochs to run')
parser.add_argument('--parallel', action ='store_true', default=False, help='use multiple GPUs for training')
parser.add_argument('--penalty', type=float, default=-0.5, help='to penalize the PN for incorrect predictions')
parser.add_argument('--alpha', type=float, default=0.8, help='probability bounding factor')
parser.add_argument('--lr_size', type=int, default=32, help='Policy Network Image Size')
parser.add_argument('--test_interval', type=int, default=10, help='At what epoch to test the model')
args = parser.parse_args()

if not os.path.exists(args.cv_dir):
    os.system('mkdir ' + args.cv_dir)
utils.save_args(__file__, args)

IMG_PATH = 'data/images'
MSK_PATH = 'data/voting_masks'
TH_FIRE = 0.25 # fire threshold

def test_unet():
    img_path = os.path.join('data/images', 'LC08_L1GT_142045_20200808_20200808_01_RT_p00533.tif')
    from Seg_generator import get_img_762bands
    img3c = torch.from_numpy(get_img_762bands(img_path)) # in 3 channels
    # y_pred = keras_unet.predict(np.array([img3c]), batch_size=1)
    img3c = torch.unsqueeze(img3c, 0).permute(0, 3, 1, 2)
    y_pred = pytorch_unet.forward(img3c)
    visualize_with_seg_mask(img3c[0], y_pred[0])
    # sum(sum(y_pred[0][0]>0))

def get_SegNet_prediction(images):
    # keras_unet = get_model_keras(model_name='unet',
    #                              input_height=IMAGE_SIZE[0], input_width=IMAGE_SIZE[1],
    #                              n_filters=N_FILTERS, n_channels=N_CHANNELS)
    # keras_unet.load_weights(WEIGHTS_FILE)
    # images = images.permute(0, 2, 3, 1)
    y_pred = pytorch_unet.forward(images.cpu()) #, batch_size=args.batch_size)

    # y_pred = y_pred[:, :, :, 0] > TH_FIRE # edw ginontai binary oi times
    y_pred = y_pred > TH_FIRE
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

    dice = dice_coefficient(preds, targets)
    penalty = 1-dice # the segmentation error between 0 and 1

    reward = sparse_reward - penalty

    return reward.unsqueeze(1), dice

def save_masked_img_grid(epoch, batch_idx ,inputs_sample, mode):
    # patches_dropped = []
    # for i in range(len(agent_actions)):
    #     patches_dropped.append( str(16 - sum(agent_actions[i].int()).item()) )

    # make the grid of 4 masked images
    fig, axarr = plt.subplots(2, 2)

    for ax, img in zip(axarr.ravel(), inputs_sample):
        ax.imshow(img.permute(1, 2, 0).cpu())
    fig.suptitle(mode)
    fig.savefig("action_progress/"+mode+"/Epoch" + str(epoch) +"_batch_"+ str(batch_idx+1) + ".jpg")
    plt.close("all")

def train(epoch):

    agent.train() # trains the policy network only

    rewards, rewards_baseline, policies, dice_coefs = [], [], [], []

    for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(trainloader), total=len(trainloader), disable=True): # , disable=True

        inputs = inputs.float().permute(0, 3, 1, 2)
        targets = targets.float().permute(0, 3, 1, 2)
        # if args.parallel:
        # inputs = inputs.cuda()
        # targets = targets.cuda()

        inputs_agent = inputs.clone()
        inputs_baseline = inputs.clone()
        inputs_sample = inputs.clone()

        # Run the low-res image through Policy Network
        inputs_agent = torch.nn.functional.interpolate(inputs_agent, (args.lr_size, args.lr_size))

        outputs_agent = agent.forward(inputs_agent, 'R_Landsat8', 'lr')
        probs = torch.sigmoid(outputs_agent)
        probs = args.alpha * probs + (1-args.alpha) * (1-probs) # temperature scaling (to encourage exploration)

        # --Got one prob vector for each image of the batch--

        distr = Bernoulli(probs) # batch_size x patch_size

        agent_actions = distr.sample()

        # Test time policy - used as baseline policy in the training step
        baseline_actions = probs.data.clone()
        baseline_actions[baseline_actions<0.5] = 0.0
        baseline_actions[baseline_actions>=0.5] = 1.0

        # Agent sampled high resolution images
        inputs_baseline = utils.get_agent_masked_image(inputs_baseline, baseline_actions, mappings, patch_size)

        inputs_sample = utils.get_agent_masked_image(inputs_sample, agent_actions.int(), mappings, patch_size)
        # get_agent_masked_image(): get patches of input based on policy arg 2

        # --POLICY NETWORK DONE--

        # for i in range(5):
        #     plt.imshow( inputs_sample[i].cpu().permute(1, 2, 0) )
        #     plt.show()

        # Input (masked HR) images to the Segmentation Network

        # baseline_vs_agent_sampling(inputs[0], inputs_baseline[0].cpu(), inputs_sample[0].cpu())
        # baseline_vs_agent_sampling(inputs[1], inputs_baseline[1].cpu(), inputs_sample[1].cpu())
        preds_baseline = get_SegNet_prediction(inputs_baseline)
        preds_sample = get_SegNet_prediction(inputs_sample)

        # Compute reward for baseline and sampled policy
        preds_baseline = torch.from_numpy(preds_baseline)
        preds_sample = torch.from_numpy(preds_sample)
        reward_baseline, dice_baseline = compute_SegNet_reward(preds_baseline, targets, baseline_actions.data)
        reward_sample, dice = compute_SegNet_reward(preds_sample, targets, agent_actions.data)
        # advantage = reward_sample.cuda().float() - reward_baseline.cuda().float()
        advantage = reward_sample.float() - reward_baseline.float()
        # print("adv:", advantage) # batch_size x 1 -> 1 adv. for each image

        # Find the loss for only the policy network
        loss = -distr.log_prob(agent_actions)  # formula (9)
        loss = loss * advantage.expand_as(agent_actions) # REINFORCE
        # print(loss.size()) # batch_size x patch_size -> 1 loss for each patch of each image
        loss = loss.mean() # negative value for the loss is normal in policy gradient (this loss is useless in terms of performance)
        # why take the mean of the losses of the 16 agent actions? h seira den paizei rolo?
        optimizer.zero_grad()
        loss.backward() # compute gradients of loss w.r.t. weights of policy network
        optimizer.step() # run Adam optimizer to update the weights

        rewards.append(reward_sample.cpu())
        rewards_baseline.append(reward_baseline.cpu())
        policies.append(agent_actions.data.cpu())
        dice_coefs.append(dice)

        # Save the final states (one epoch, 16 images)
        if epoch % 10 == 0:
            dropped = str(16-sum(agent_actions[0].int()).item()) # at zero position is the only one image
            # save_masked_img_grid(epoch, batch_idx, inputs_sample, "training")
            plt.imsave("action_progress/Epoch" + str(epoch) + "_patches_dropped_" + dropped + ".jpg",
                       inputs_sample[0].permute(1, 2, 0).cpu().numpy())

    reward, sparsity, variance, policy_set, avg_dice = utils.performance_stats(policies, rewards, dice_coefs)
    # to sparsity einai posa patches kata meso oro epilegei
    exp_return.append(reward)
    # torch.cuda.empty_cache()
    print('Train: %d | Rw: %.3f | Dice: %.3f | S: %.3f'%(epoch, reward, avg_dice, sparsity))

    # log_value('train_accuracy', accuracy, epoch)
    # log_value('train_reward', reward, epoch)
    # log_value('train_sparsity', sparsity, epoch)
    # log_value('train_variance', variance, epoch)
    # log_value('train_baseline_reward', torch.cat(rewards_baseline, 0).mean(), epoch)
    # log_value('train_unique_policies', len(policy_set), epoch)

def test(epoch):

    agent.eval() # flag: deactivate training (gradient update) mode

    rewards, policies, dice_coef = [], [], []
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
        inputs = utils.get_agent_masked_image(inputs, policy, mappings, patch_size)
        preds = get_SegNet_prediction(inputs)

        preds = torch.from_numpy(preds)
        reward, dice = compute_SegNet_reward(preds, targets, policy.data)

        rewards.append(reward)
        policies.append(policy.data)
        dice_coef.append(dice)

        save_masked_img_grid(epoch, batch_idx, inputs, "test")

    reward, sparsity, variance, policy_set, dice = utils.performance_stats(policies, rewards, dice_coef)
    torch.cuda.empty_cache()

    print('Test - Rw: %.3f | Dice: %.3f | S: %.3f | V: %.3f\n'%(reward, dice, sparsity, variance))
    # log_value('test_accuracy', accuracy, epoch)
    # log_value('test_reward', reward, epoch)
    # log_value('test_sparsity', sparsity, epoch)
    # log_value('test_variance', variance, epoch)
    # log_value('test_unique_policies', len(policy_set), epoch)

    # Save the Policy Network - Classifier is fixed in this phase
    agent_state_dict = agent.module.state_dict() if args.parallel else agent.state_dict()
    state = {
      'agent': agent_state_dict,
      'epoch': epoch,
      'reward': reward,
      'dice': dice
    }
    # torch.save(state, args.cv_dir+'Landsat-8/Policy_ckpt_E_%d_dice_%.3f_R_%.3f'%(epoch, dice, reward))


#--------------------------------------------------------------------------------------------------------#
trainset, testset = utils.get_dataset(args.model, args.data_dir)
# train = trainset.__getitem__(0)
# test = testset.__getitem__(0)
# visualize_images(trainset.data, trainset.targets, "train sample")
# visualize_images(testset.data, testset.targets, "test sample")
trainloader = torchdata.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
testloader = torchdata.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0)

# 1. Load the Agent
agent = utils.get_model('Landsat8_ResNet')
print('PN loaded!')

# Save the args to the checkpoint directory
# configure(args.cv_dir+'/log', flush_secs=5)

# Agent Action Space
mappings, _, patch_size = utils.action_space_model('Landsat-8')

# 2. Load the segmentation network (Unet-Light-3c) --> den einai to light 3c (malakes)

pytorch_unet = get_model_pytorch(model_name='unet', n_filters=N_FILTERS, n_channels=N_CHANNELS)
# summary(pytorch_unet, (3, 256, 256))

print('U-Net loaded!')

# 3. Load the weights (trained on voting scheme)
# keras_unet.load_weights(WEIGHTS_FILE)
pytorch_unet.load_state_dict(torch.load('cv/tmp/Landsat-8/unet/pytorch_unet.pt'))
pytorch_unet.eval()
print('U-Net weights loaded!')

# print("Keras:", keras_unet.weights[0].shape, " PyTorch:", pytorch_unet.down1.conv_block[0].weight.shape)

# Load the Policy Network (if checkpoint exists)
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

exp_return = []
start_epoch = 1

# OTAN TO TREXEIS ME > 1 image sto batch EPANEFERE TO BATCHNORM
for epoch in range(start_epoch, start_epoch+args.max_epochs):
    train(epoch)
    if epoch % args.test_interval == 0:
        test(epoch)

# exp_return = sum(exp_return)/len(exp_return)
torch.save(agent.state_dict(), f"checkpoints/PN_{len(trainset)}_train_images_{args.max_epochs}_epochs.pt")

plt.figure()
plt.semilogy(exp_return)
plt.xlabel('Epochs')
plt.ylabel('Expected return')
# plt.xlim([0, len(losses)])
plt.grid(linestyle=':', which='both')
plt.show()