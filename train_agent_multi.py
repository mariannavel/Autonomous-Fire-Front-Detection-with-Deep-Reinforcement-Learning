import os
import random

import torch
import torch.nn.functional as F
import torch.utils.data as torchdata
import tqdm
import torch.optim as optim
from torch.distributions import Bernoulli, Categorical, Multinomial
from utils import agent_utils
from utils.custom_dataloader import LandsatDataset
from utils.unet_utils import *
from tensorboard_logger import configure
import torch.backends.cudnn as cudnn
import time
import matplotlib.pyplot as plt

random.seed(42)
cudnn.benchmark = True
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
K = 16
import argparse
parser = argparse.ArgumentParser(description='Policy Network Training')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--model', default='ResNet_Landsat8')
parser.add_argument('--data_dir', default=f'data/toy_dset/')
parser.add_argument('--load', default=None, help='checkpoint to load pretrained agent')
parser.add_argument('--cv_dir', default=f'train_multi/15/')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--max_epochs', type=int, default=100, help='total epochs to run')
parser.add_argument('--alpha', type=float, default=0.8, help='probability bounding factor')
parser.add_argument('--penalty', type=int, default=0.5, help="agent's penalty for bad selection")
parser.add_argument('--LR_size', type=int, default=32, help='Policy Network Image Size')
parser.add_argument('--test_interval', type=int, default=10)
parser.add_argument('--ckpt_interval', type=int, default=100)
args = parser.parse_args()

if not os.path.exists(args.cv_dir + '/checkpoints'):
    os.makedirs(args.cv_dir + '/checkpoints')

agent_utils.save_args(__file__, args)

def visualize_image(set, title=''):
    # input ndarray: 256 x 256 x 3
    for image in set:
        plt.imshow(image)
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

def visualize_images(img1, img2, title="", savepath=""):
    # input ndarray: n x 256 x 256 x 3
    for i, (img, msk) in enumerate(zip(img1, img2)):
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(msk)
        plt.axis('off')
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

class Agent:
    def __init__(self, state, action, model):
        self.state = state
        self.action = action
        self.model = model

    def step(self):
        pass

def transitions_next(p_prev, A):
    """
    Given a vector of probabilities (p_prev) and a binary action array (A) it
    returns a vector of probabilities over the corresponding zero elements of
    A, and zeros in the positions where A is 1.

    return: new multinomial distribution over one patch less
    """

    p_cur = (A^1).float() # complement with XOR
    inter_distr = p_cur * p_prev # element-wise

    nonzero_distr = inter_distr[inter_distr != 0]

    # get probabilistic values
    prob_distr = F.softmax(nonzero_distr, dim=0)
    i = 0
    for val in p_cur: # rescale prob_distr
        if val == 1:
            val *= prob_distr[i]
            i += 1
    return p_cur

    # new_distr = torch.zeros(prev_distr.shape[0], prev_distr.shape[1]-1)
    # new_distr = prev_distr.detach().clone()

    # for i, probs in enumerate(prev_distr):
        # Eliminate previously selected action index
        # new_distr[i] = torch.cat([probs[0:actions[i]], probs[actions[i] + 1:]])
    # for i in range(new_distr.shape[0]):
    #     new_distr[i, actions[i]] = 0
        # Reformulate vector
        # new_distr[i] = F.softmax(new_distr[i], dim=1)

def train(epoch):

    exp_rewards, action_set, dice_coefs = [], [], []

    for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(trainloader), disable=True):

        inputs = inputs.to(device)
        targets = targets.to(device)

        inputs, inputs_agent, targets = agent_utils.get_input_lr(args.LR_size, inputs, targets)

        # 1. Input LR image batch to Policy Network
        logits = agent.forward(inputs_agent, activation=None) # softmax: sum(probs)==1
        probs = F.softmax(logits, dim=1)
        # probs = args.alpha * probs + (1-args.alpha) * (1-probs) # temperature scaling
        policy = Multinomial(total_count=1, probs=probs, validate_args=False)

        A = torch.zeros(args.batch_size, K).to(device) # initialize action array

        Rewards = [[] for _ in range(args.batch_size)] # batch_size vectors of DC

        # ** Start episode **
        for _ in range(4):  # max int(K/2) time steps

            # 2. Sample patch from multinomial distribution
            distr = Multinomial(total_count=1, probs=probs, validate_args=False)  # equivalent to torch.multinomial

            actions = distr.sample()

            # one_hot_actions = F.one_hot(actions.to(torch.int64), K)
            A += actions

            # 3. Apply mask to x_hr
            x_hr_m = agent_utils.get_agent_masked_image(inputs, A, mappings, patch_size)
            visualize_images(x_hr_m.permute(0,2,3,1).cpu(), targets.permute(0,2,3,1).cpu())

            # 4. Input x_h^m to U-Net
            preds = get_prediction(x_hr_m.cpu(), pytorch_unet, device) # get segmentation mask

            for i in range(args.batch_size):
                if i > 0 and len(Rewards[i]) != 0 and Rewards[i][-1] > 0.9: # stop sampling from that image
                    continue
                # 5. Evaluate Dice Coefficient
                DC = dice_coefficient(preds[i], targets[i])

                if len(Rewards[i]) != 0 and DC <= max(Rewards[i]):
                    DC -= args.penalty
                Rewards[i].append(DC.item())

                probs[i] = transitions_next(probs[i].detach(), A[i].int())

        # a vector of rewards for each image --> evaluate Utility
        # if rewards in increasing order: E[U] = 1 --> not?
        E_U = [sum(R)/len(R) for R in Rewards] # en expected utility for each image

        loss = -policy.log_prob(A) # giati exoun aftes tis times???
        loss = loss * torch.Tensor(E_U).to(device) # REINFORCE
        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        exp_rewards.append(E_U)
        action_set.append(A.data.cpu())

    # avg_reward, avg_dc, sparsity, variance = agent_utils.get_performance_stats(action_set, rewards, dice_coefs, train_stats)
    # avg_rw_baseline = torch.cat(rewards_baseline, 0).mean()
    # train_stats["reward_baseline"].append(avg_rw_baseline)
    #
    # t = time.time()-start_time
    # print('Train: %d | Rw: %.3f | Dice: %.3f | S: %.3f | V: %.3f | %.3f s'%(epoch, avg_reward,
    #         avg_dc, sparsity, variance, t))

    # utils.log_value('train_baseline_reward', avg_rw_baseline, epoch)
    # utils.save_logs(epoch, avg_reward, avg_dc, sparsity, variance, mode="train")

def test(epoch):

    # agent.eval() # flag: deactivate training (gradient update) mode

    rewards, action_set, dice_coef = [], [], []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(testloader), total=len(testloader), disable=True):

            inputs.to(device)
            targets.to(device)

            inputs, inputs_agent, targets = agent_utils.get_input_lr(args.LR_size, inputs, targets)

            probs = agent.forward(inputs_agent.to(device))

            # Sample the test-time policy
            actions = probs.data.clone()
            actions[actions<0.5] = 0.0
            actions[actions>=0.5] = 1.0

            # Get the masked high-res image and perform inference
            masked_images = agent_utils.get_agent_masked_image(inputs, actions, mappings, patch_size)
            preds = get_prediction(masked_images.cpu(), pytorch_unet, device)

            reward, dice = compute_reward(preds.cpu(), targets, actions.data, device)

            rewards.append(reward)
            action_set.append(actions.data)
            dice_coef.append(dice)

            torch.cuda.empty_cache()

    # agent_utils.save_masked_img_grid(epoch, batch_idx, inputs, "validation")

    avg_reward, avg_dc, sparsity, variance = agent_utils.get_performance_stats(action_set, rewards, dice_coef, test_stats)

    print('Test | Rw: %.3f | Dice: %.3f | S: %.3f | V: %.3f\n'%(avg_reward, avg_dc, sparsity, variance))

    agent_utils.save_logs(epoch, avg_reward, avg_dc, sparsity, variance, mode="test")
    if epoch % args.ckpt_interval == 0:
        agent_utils.save_agent_model(epoch, args, agent, avg_reward, avg_dc)
        agent_utils.save_stats(args.cv_dir, train_stats, test_stats)

#--------------------------------------------------------------------------------------------------------#
trainset = LandsatDataset(args.data_dir + 'train.pkl')
testset = LandsatDataset(args.data_dir + 'test.pkl')

# visualize_images(trainset.data, trainset.targets, "train sample")
# visualize_images(testset.data, testset.targets, "test sample")
trainloader = torchdata.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=0) # SHUFFLE WAS TRUE !!! :\
testloader = torchdata.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load the Agent
agent = agent_utils.get_model(args.model)
print('Agent loaded')

# Save the log values to the checkpoint directory
configure(args.cv_dir+'/logs', flush_secs=5)

# Agent Action Space
mappings, _, patch_size = agent_utils.get_action_space('Landsat8')

# 2. Load the segmentation network (Unet-Light)
pytorch_unet = get_model_pytorch(model_name='unet', n_filters=16, n_channels=3)
# summary(pytorch_unet, (3, 256, 256))
print('U-Net loaded')

# 3. Load the weights (trained on voting scheme)
pytorch_unet.load_state_dict(torch.load(agent_utils.CKPT_UNET))
pytorch_unet.eval() # U-Net must be on cpu, else CUDA out of memory
print('U-Net weights loaded')

start_epoch = 1
# Load the Policy Network (if checkpoint exists)
if args.load is not None:
    checkpoint = torch.load(args.load)
    agent.load_state_dict(checkpoint['agent'])
    start_epoch = checkpoint['epoch'] + 1
    print("Pretrained agent loaded")

agent.to(device) # next(agent.parameters()).is_cuda

optimizer = optim.Adam(agent.parameters(), lr=args.lr)

# Store the Expected Value of each statistic for every epoch
train_stats = {"return": [], "dice": [], "sparsity": [], "variance": [], "reward_baseline": []}
test_stats = {"return": [], "dice": [], "sparsity": [], "variance": []}

start_time = time.time()

for epoch in range(start_epoch, args.max_epochs+1):
    train(epoch)
    # if epoch % args.test_interval == 0:
    #     test(epoch)

print('Runtime (min): %.2f' % ((time.time()-start_time)/60.))