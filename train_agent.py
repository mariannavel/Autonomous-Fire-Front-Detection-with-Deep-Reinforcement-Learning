"""
This file trains the policy network using the U-Net architecture
as a policy evaluation step.
To Run on Different Benchmarks:
    python train_agent.py --model ResNet_Landsat8, ResNet18_Landsat8 CNN_Landsat8
       --lr 1e-4
       --cv_dir configuration directory
       --batch_size 1048 (Higher is better)
       --LR_size 8, 56 (Depends on the dataset)
"""
import os
import torch.nn as nn
# Autograd automatically supports Tensors with requires_grad set to True
import torch.utils.data as torchdata
import tqdm
import torch.optim as optim
from torch.distributions import Bernoulli
from utils import utils
from utils.custom_dataloader import LandsatDataset
from unet.unet_models_pytorch import get_model_pytorch
# from torchsummary import summary
from tensorboard_logger import configure
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import time
from segnet import *

cudnn.benchmark = True

NUM_SAMPLES = 512

import argparse
parser = argparse.ArgumentParser(description='Policy Network Training')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--model', default='ResNet_Landsat8', help='R<depth>_<dataset> see utils.py for a list of configurations')
parser.add_argument('--data_dir', default=f'data/{NUM_SAMPLES}/stratified/', help='data directory')
parser.add_argument('--load', default=None, help='checkpoint to load pretrained agent')
parser.add_argument('--cv_dir', default=f'train_agent/{NUM_SAMPLES}/stratified/Res/', help='configuration directory (models and logs are saved here)')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--max_epochs', type=int, default=2000, help='total epochs to run')
parser.add_argument('--alpha', type=float, default=0.8, help='probability bounding factor')
parser.add_argument('--LR_size', type=int, default=32, help='Policy Network Image Size')
parser.add_argument('--test_interval', type=int, default=50, help='Every how many epoch to test the model')
parser.add_argument('--ckpt_interval', type=int, default=100, help='Every how many epoch to save the model')
args = parser.parse_args()

if not os.path.exists(args.cv_dir + '/checkpoints'):
    os.makedirs(args.cv_dir + '/checkpoints')
utils.save_args(__file__, args)

def train(epoch):

    agent.train()  # trains the policy network only

    rewards, rewards_baseline, action_set, dice_coefs = [], [], [], []

    for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(trainloader), total=len(trainloader), disable=True):

        inputs = inputs.to(device)
        targets = targets.to(device)

        inputs, inputs_agent, targets = utils.preprocess_inputs(args.LR_size, inputs, targets)

        # Run LR image through Policy Network
        probs = agent.forward(inputs_agent)
        probs = args.alpha * probs + (1-args.alpha) * (1-probs) # temperature scaling (to encourage exploration)

        # --Got one prob vector for each image of the batch--

        policy = Bernoulli(probs) # batch_size x patch_size

        agent_actions = policy.sample()

        # Test time policy - used as baseline policy in the training step
        baseline_actions = probs.data.clone()
        baseline_actions[baseline_actions<0.5] = 0.0
        baseline_actions[baseline_actions>=0.5] = 1.0

        # --POLICY NETWORK DONE--

        # Agent sampled high resolution images
        maskedHR_baseline = utils.get_agent_masked_image(inputs, baseline_actions, mappings, patch_size)
        maskedHR_sampled = utils.get_agent_masked_image(inputs, agent_actions.int(), mappings, patch_size)
        # get_agent_masked_image(): get patches of input based on policy arg 2

        # Input (masked HR) images to the Segmentation Network

        preds_baseline = get_unet_prediction(maskedHR_baseline.cpu(), pytorch_unet, device)
        preds_sample = get_unet_prediction(maskedHR_sampled.cpu(), pytorch_unet, device)

        # Compute reward for baseline and sampled policy
        reward_baseline, _ = compute_SegNet_reward(preds_baseline, targets, baseline_actions.data, device)
        reward_sample, dice = compute_SegNet_reward(preds_sample, targets, agent_actions.data, device)
        # advantage = reward_sample.cuda().float() - reward_baseline.cuda().float()
        advantage = reward_sample.float() - reward_baseline.float()
        # print("adv:", advantage) # batch_size x 1 -> 1 adv. for each image

        # Find the loss for only the policy network
        loss = -policy.log_prob(agent_actions)  # formula (9)
        loss = loss * advantage.expand_as(agent_actions) # REINFORCE
        # print(loss.size()) # batch_size x patch_size -> 1 loss for each patch of each image
        loss = loss.mean() # negative value for the loss is normal in policy gradient (this loss is useless in terms of performance)
        # In Policy Optimization we don't want to craft specific choices of actions, but compute a cumulative reward for
        # all the actions the agent takes and give it a feedback for its actions overall (as a feedback for a whole game round).
        optimizer.zero_grad()
        loss.backward() # compute gradients of loss w.r.t. weights of policy network
        optimizer.step() # run Adam optimizer to update the weights

        rewards.append(reward_sample.cpu())
        rewards_baseline.append(reward_baseline.cpu())
        action_set.append(agent_actions.data.cpu())
        dice_coefs.append(dice.cpu())

        torch.cuda.empty_cache()

        # Save the final states (one epoch, 16 images)
        # if epoch % 10 == 0:
            # dropped = str(16-sum(agent_actions[0].int()).item()) # at zero position is the only one image
            # plt.imsave("action_progress/Epoch" + str(epoch) + "_patches_dropped_" + dropped + ".jpg",
            #            maskedHR_sampled[0].permute(1, 2, 0).cpu().numpy())

    avg_reward, avg_dc, sparsity, variance = utils.get_performance_stats(action_set, rewards, dice_coefs, train_stats)
    avg_rw_baseline = torch.cat(rewards_baseline, 0).mean()
    train_stats["reward_baseline"].append(avg_rw_baseline)

    t = time.time()-start_time
    print('Train: %d | Rw: %.3f | Dice: %.3f | S: %.3f | V: %.3f | %.3f s'%(epoch, avg_reward,
            avg_dc, sparsity, variance, t))

    utils.log_value('train_baseline_reward', avg_rw_baseline, epoch)
    utils.save_logs(epoch, avg_reward, avg_dc, sparsity, variance, mode="train")

def test(epoch):

    # agent.eval() # flag: deactivate training (gradient update) mode

    rewards, action_set, dice_coef = [], [], []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(testloader), total=len(testloader), disable=True):

            inputs.to(device)
            targets.to(device)

            inputs, inputs_agent, targets = utils.preprocess_inputs(args.LR_size, inputs, targets)

            probs = agent.forward(inputs_agent.to(device))

            # Sample the test-time policy
            actions = probs.data.clone()
            actions[actions<0.5] = 0.0
            actions[actions>=0.5] = 1.0

            # Get the masked high-res image and perform inference
            masked_images = utils.get_agent_masked_image(inputs, actions, mappings, patch_size)
            preds = get_unet_prediction(masked_images.cpu(), pytorch_unet, device)

            reward, dice = compute_SegNet_reward(preds.cpu(), targets, actions.data, device)

            rewards.append(reward)
            action_set.append(actions.data)
            dice_coef.append(dice)

            torch.cuda.empty_cache()

    # utils.save_masked_img_grid(epoch, batch_idx, inputs, "validation")

    avg_reward, avg_dc, sparsity, variance = utils.get_performance_stats(action_set, rewards, dice_coef, test_stats)

    print('Test | Rw: %.3f | Dice: %.3f | S: %.3f | V: %.3f\n'%(avg_reward, avg_dc, sparsity, variance))

    utils.save_logs(epoch, avg_reward, avg_dc, sparsity, variance, mode="test")
    if epoch % args.ckpt_interval == 0:
        utils.save_agent_model(epoch, args, agent, avg_reward, avg_dc)
        utils.save_stats(args.cv_dir, train_stats, test_stats, num_samples=NUM_SAMPLES)

#--------------------------------------------------------------------------------------------------------#
trainset = LandsatDataset(args.data_dir + 'train.pkl')
testset = LandsatDataset(args.data_dir + 'test.pkl')

# visualize_images(trainset.data, trainset.targets, "train sample")
# visualize_images(testset.data, testset.targets, "test sample")
trainloader = torchdata.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
testloader = torchdata.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load the Agent
agent = utils.get_model(args.model)
print('Agent loaded')
# summary(agent, (3, 256, 256))

# Save the log values to the checkpoint directory
configure(args.cv_dir+'/logs', flush_secs=5)

# Agent Action Space
mappings, _, patch_size = utils.action_space_model('Landsat8')

# 2. Load the segmentation network (Unet-Light)
pytorch_unet = get_model_pytorch(model_name='unet', n_filters=16, n_channels=3)
# summary(pytorch_unet, (3, 256, 256))
# print('U-Net loaded')
# exit(0)

# 3. Load the weights (trained on voting scheme)
pytorch_unet.load_state_dict(torch.load(utils.CKPT_UNET))
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
if torch.cuda.device_count() > 1:
    agent = nn.DataParallel(agent)
    pytorch_unet = nn.DataParallel(pytorch_unet)
# torch.cuda.device_count() == 1 ---> can't use GPU parallelization

agent.to(device) # next(agent.parameters()).is_cuda

optimizer = optim.Adam(agent.parameters(), lr=args.lr)

# Store the Expected Value of each statistic for every epoch
train_stats = {"return": [], "dice": [], "sparsity": [], "variance": [], "reward_baseline": []}
test_stats = {"return": [], "dice": [], "sparsity": [], "variance": []}

start_time = time.time()

for epoch in range(start_epoch, args.max_epochs+1):
    train(epoch)
    if epoch % args.test_interval == 0:
        test(epoch)

print('Runtime (min): %.2f' % ((time.time()-start_time)/60.))