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
import torch
# Autograd automatically supports Tensors with requires_grad set to True !!!
import torch.utils.data as torchdata
import matplotlib.pyplot as plt
import tqdm
import torch.optim as optim

from torch.distributions import Bernoulli
from utils import utils
from SegNet.unet_models_pytorch import get_model_pytorch
# from torchsummary import summary
from tensorboard_logger import configure
import torch.backends.cudnn as cudnn
import time
from visualize import visualize_images

cudnn.benchmark = True

WEIGHTS_FILE ='cv/tmp/Landsat-8/unet/model_unet_voting_final_weights.h5'

import argparse
parser = argparse.ArgumentParser(description='Policy Network Training')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate') # DECREASED lr 0.01 --> 0.001
parser.add_argument('--model', default='ResNet_Landsat8', help='R<depth>_<dataset> see utils.py for a list of configurations')
parser.add_argument('--ckpt_hr_cl', help='checkpoint directory for the high resolution classifier')
parser.add_argument('--data_dir', default='data/', help='data directory')
parser.add_argument('--load', default=None, help='checkpoint to load agent from')
parser.add_argument('--cv_dir', default='cv/tmp/', help='checkpoint directory (models and logs are saved here)')
parser.add_argument('--batch_size', type=int, default=32, help='batch size') # INCREASED batch size 8 --> 16 --> 32
parser.add_argument('--max_epochs', type=int, default=1000, help='total epochs to run')
parser.add_argument('--parallel', action ='store_true', default=False, help='use multiple GPUs for training')
parser.add_argument('--penalty', type=float, default=-0.5, help='to penalize the PN for incorrect predictions')
parser.add_argument('--alpha', type=float, default=0.8, help='probability bounding factor')
parser.add_argument('--lr_size', type=int, default=32, help='Policy Network Image Size')
parser.add_argument('--test_interval', type=int, default=10, help='Every how many epoch to test the model')
parser.add_argument('--ckpt_interval', type=int, default=100, help='Every how many epoch to save the model')
args = parser.parse_args()

if not os.path.exists(args.cv_dir):
    os.system('mkdir ' + args.cv_dir)
utils.save_args(__file__, args)

def dice_coefficient(predicted, target, smooth=1):
    """
    :param predicted: the predicted segmentation mask
    :param smooth: Smoothing factor (in [0, 1] to avoid division by zero
    :return: the dice coefficient between predicted and target
    """
    dice = torch.zeros(predicted.shape[0])
    # iterate a batch of segmentation masks
    for i, (p, t) in enumerate(zip(predicted, target)):
        intersection = torch.sum(p * t)
        union = torch.sum(p) + torch.sum(t)
        dice[i] = (2. * intersection + smooth) / (union + smooth)
    return dice

def compute_SegNet_reward(preds, targets, policy):
    """
    :param preds: the predicted segmentation masks
    :param targets: the ground truth segmentation masks
    :param policy: binary vector indicating sampled or non-sampled patches
    """

    patch_use = policy.sum(1).float() / policy.size(1) # I will penalize the policy with the number of sampled patches
    sparse_reward = 1.0 - patch_use**2

    dice = dice_coefficient(preds, targets) # batch_size x 1
    for dc in dice: # assert dc >= 0 and dc <= 1
        torch._assert(dc >= 0 and dc <= 1, "Dice coefficient out of range [0,1] !")

    penalty = 1-dice # the segmentation error between 0 and 1

    reward = sparse_reward - penalty.to(device)

    return reward.unsqueeze(1), dice

def get_SegNet_prediction(images, unet):
    TH_FIRE = 0.25
    # keras_unet = get_model_keras(model_name='unet',
    #                              input_height=IMAGE_SIZE[0], input_width=IMAGE_SIZE[1],
    #                              n_filters=N_FILTERS, n_channels=N_CHANNELS)
    # keras_unet.load_weights(WEIGHTS_FILE)
    y_pred = unet.forward(images) #, batch_size=args.batch_size)
    # y_pred = y_pred[:, :, :, 0] > TH_FIRE # edw ginontai binary oi times
    y_pred = y_pred > TH_FIRE
    # pred_masks = np.array(y_pred * 255, dtype=np.uint8)
    return (y_pred*1).to(device)

def train(epoch):

    agent.train()  # trains the policy network only

    rewards, rewards_baseline, policies, dice_coefs = [], [], [], []

    for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(trainloader), total=len(trainloader), disable=True):

        inputs = inputs.float().permute(0, 3, 1, 2)
        targets = targets.float().permute(0, 3, 1, 2)
        # if args.parallel:
        inputs = inputs.to(device)
        targets = targets.to(device)

        inputs_agent = inputs.clone()
        inputs_baseline = inputs.clone()
        inputs_sample = inputs.clone()

        inputs_agent = torch.nn.functional.interpolate(inputs_agent, (args.lr_size, args.lr_size))

        # Run the LR image through the Policy Network
        probs = agent.forward(inputs_agent)
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

        # Input (masked HR) images to the Segmentation Network

        preds_baseline = get_SegNet_prediction(inputs_baseline.cpu(), pytorch_unet)
        preds_sample = get_SegNet_prediction(inputs_sample.cpu(), pytorch_unet)

        # Compute reward for baseline and sampled policy
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
        dice_coefs.append(dice.cpu())
        torch.cuda.empty_cache()

        # Save the final states (one epoch, 16 images)
        # if epoch % 10 == 0:
            # dropped = str(16-sum(agent_actions[0].int()).item()) # at zero position is the only one image
            # plt.imsave("action_progress/Epoch" + str(epoch) + "_patches_dropped_" + dropped + ".jpg",
            #            inputs_sample[0].permute(1, 2, 0).cpu().numpy())

    E = {}  # stores the Expected Value of each statistic
    E["return"], E["dice"], E["sparsity"], E["variance"] = utils.performance_stats(policies, rewards, dice_coefs)
    E["rewards_baseline"] = rewards_baseline
    exp_return.append(E["return"])
    t = time.time()-start_time
    print('Train: %d | Rw: %.3f | Dice: %.3f | S: %.3f | V: %.3f | %.3f s'%(epoch, E["return"], E["dice"], E["sparsity"], E["variance"], t))

    utils.save_logs(epoch, E, mode="train")


def test(epoch):

    agent.eval() # flag: deactivate training (gradient update) mode

    rewards, policies, dice_coef = [], [], []
    for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(testloader), total=len(testloader)):
        # UserWarning: volatile was removed and now has no effect. Use `with torch.no_grad():` instead
        with torch.no_grad():
            inputs = inputs.float().permute(0, 3, 1, 2)
            targets = targets.float().permute(0, 3, 1, 2)
            # targets = torch.Tensor([int(val) for val in targets])

            # if args.parallel:
            inputs.to(device)
            targets.to(device)

            # Get the low resolution agent images
            inputs_agent = inputs.clone()
            inputs_agent = torch.nn.functional.interpolate(inputs_agent, (args.lr_size, args.lr_size))
            # probs = torch.sigmoid(agent.forward(inputs_agent, 'R_Landsat8', 'lr'))

            probs = agent.forward(inputs_agent.to(device))

            # Sample the test-time policy
            policy = probs.data.clone()
            policy[policy<0.5] = 0.0
            policy[policy>=0.5] = 1.0

            # Get the masked high-res image and perform inference
            inputs = utils.get_agent_masked_image(inputs, policy, mappings, patch_size)
            preds = get_SegNet_prediction(inputs.cpu(), pytorch_unet)

        # preds = torch.from_numpy(preds)
        reward, dice = compute_SegNet_reward(preds.cpu(), targets, policy.data)

        rewards.append(reward)
        policies.append(policy.data)
        dice_coef.append(dice)
        torch.cuda.empty_cache()

        # utils.save_masked_img_grid(epoch, batch_idx, inputs, "validation")
    E = {} # stores the Expected Value of each statistic
    E["return"], E["dice"], E["sparsity"], E["variance"] = utils.performance_stats(policies, rewards, dice_coef)

    print('Test - Rw: %.3f | Dice: %.3f | S: %.3f | V: %.3f\n'%(E["return"], E["dice"], E["sparsity"], E["variance"]))

    utils.save_logs(epoch, E, mode="test")

    if epoch % args.ckpt_interval == 0:
        # save the agent model
        agent_state_dict = agent.module.state_dict() if args.parallel else agent.state_dict()
        state = {
          'agent': agent_state_dict,
          'epoch': epoch,
          'reward': E["return"],
          'dice': E["dice"]
        }
        torch.save(state, 'checkpoints/Policy_ckpt_E_%d_R_%.3f_%s'%(epoch, E["return"], args.model[0:3]))

#--------------------------------------------------------------------------------------------------------#
trainset, testset = utils.get_dataset(args.model, args.data_dir)

# visualize_images(trainset.data, trainset.targets, "train sample")
# visualize_images(testset.data, testset.targets, "test sample")
trainloader = torchdata.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
testloader = torchdata.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0) # NUM WORKERS !!!!

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load the Agent
agent = utils.get_model('ResNet_Landsat8')
agent.to(device)
# next(agent.parameters()).is_cuda

# Save the log values to the checkpoint directory
configure(args.cv_dir+'/logs', flush_secs=5)

# Agent Action Space
mappings, _, patch_size = utils.action_space_model('Landsat8')

# 2. Load the segmentation network (Unet-3c)

pytorch_unet = get_model_pytorch(model_name='unet', n_filters=16, n_channels=3)
# summary(pytorch_unet, (3, 256, 256))
print('U-Net loaded')

# 3. Load the weights (trained on voting scheme)
pytorch_unet.load_state_dict(torch.load('cv/tmp/Landsat-8/unet/pytorch_unet.pt'))
pytorch_unet.eval() # UNet must be on cpu, else CUDA out of memory
print('U-Net weights loaded')
# print(" PyTorch:", pytorch_unet.down1.conv_block[0].weight.shape)

start_epoch = 1
# Load the Policy Network (if checkpoint exists)
if args.load is not None:
    checkpoint = torch.load(args.load)
    agent.load_state_dict(checkpoint['agent'])
    start_epoch = checkpoint['epoch'] + 1
    print("Pretrained agent loaded")

# Parallelize the models if multiple GPUs available - Important for Large Batch Size
# if args.parallel:
#     agent = nn.DataParallel(agent)
#     unet = nn.DataParallel(unet)

optimizer = optim.Adam(agent.parameters(), lr=args.lr)

exp_return = []
start_time = time.time()

# OTAN TO TREXEIS ME > 1 image sto batch EPANEFERE TO BATCHNORM
for epoch in range(start_epoch, start_epoch+args.max_epochs):
    train(epoch)
    if epoch % args.test_interval == 0:
        test(epoch)

print(f'Expected Return: {sum(exp_return)/len(exp_return)}')
print(f'Runtime (sec): {time.time()-start_time}')
# torch.save(agent.state_dict(), f"checkpoints/Policy_ResNet_{len(trainset)}_train_images_{args.max_epochs+start_epoch-1}_epochs.pt")

plt.figure()
plt.semilogy(exp_return)
plt.xlabel('Epochs')
plt.title('Expected return')
plt.grid(linestyle=':', which='both')
plt.show()