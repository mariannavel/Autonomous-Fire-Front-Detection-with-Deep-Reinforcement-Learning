import os
import torch
import torchvision.transforms as transforms
from tensorboard_logger import log_value
from models import policy_net
import pickle
import shutil
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def save_args(__file__, args):
    shutil.copy(os.path.basename(__file__), args.cv_dir)
    with open(args.cv_dir+'/args.txt','w') as f:
        f.write(str(args))

def save_stats(savedir, train_stats, test_stats, num_samples):
    with open(f"{savedir}train{num_samples}", "wb") as fp:
        pickle.dump(train_stats, fp)
    with open(f"{savedir}test{num_samples}", "wb") as fp:
        pickle.dump(test_stats, fp)

def save_logs(epoch, avg_reward, avg_dc, sparsity, variance, mode="test"):
    """
    :param mode: train or test
    """
    log_value(f'{mode}_dice', avg_dc, epoch)
    log_value(f'{mode}_reward', avg_reward, epoch)
    log_value(f'{mode}_sparsity', sparsity, epoch)
    log_value(f'{mode}_variance', variance, epoch)
    # log_value(f'{mode}_unique_policies', len(stats["policy_set"]), epoch)

def save_image(img, path):
    plt.imshow(img)
    plt.axis(False)
    plt.savefig(path)

def save_agent_model(epoch, args, agent, reward, dice):

    agent_state_dict = agent.state_dict()
    state = {
        'agent': agent_state_dict,
        'epoch': epoch,
        'reward': reward,
        'dice': dice
    }
    torch.save(state, args.cv_dir + 'checkpoints/Policy_ckpt_E_%d_R_%.3f_%s' % (epoch, reward, args.model[0:3]))

def get_transforms():
    """
    :return: transforms for Landsat-8 train-test sets
    """

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ])

    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ])

    return transform_train, transform_test

def get_input_lr(LR_size, inputs, targets):
    inputs = inputs.float().permute(0, 3, 1, 2)
    targets = targets.float().permute(0, 3, 1, 2)
    inputs_lr = torch.nn.functional.interpolate(inputs.clone(), (LR_size, LR_size))
    return inputs, inputs_lr, targets

def get_performance_stats(actions, rewards, dc, stats_dict):
    """
    Save performance metrics from a single epoch in stats_dict.
    :param actions: [list] the binary action set
    :param rewards: [list] the rewards of each iteration
    :param dc: [list] the dice coefficient of each iteration
    :return stats to print
    """
    actions = torch.cat(actions, 0)
    rewards = torch.cat(rewards, 0) # For each sample I get one value
    dice_coefs = torch.cat(dc, 0)   # also

    avg_reward = rewards.mean()
    avg_dc = dice_coefs.mean() # high values at the beginning because agent samples more patches randomly

    sparsity = actions.sum(1).mean() # average selected patches
    variance = actions.sum(1).std()

    stats_dict["return"].append(avg_reward.item())
    stats_dict["dice"].append(avg_dc.item())
    stats_dict["sparsity"].append(sparsity.item())
    stats_dict["variance"].append(variance.item())

    return avg_reward, avg_dc, sparsity, variance

def get_agent_masked_image(input_org, policy, mappings, patch_size):
    """ Generate masked images w.r.t policy learned by the agent.
    """
    input_full = input_org.clone()
    sampled_img = torch.zeros([input_org.shape[0], input_org.shape[1], input_org.shape[2], input_org.shape[3]])
    for pl_ind in range(policy.shape[1]):
        mask = (policy[:, pl_ind] == 1).cpu()
        sampled_img[:, :, mappings[pl_ind][0]:mappings[pl_ind][0]+patch_size, mappings[pl_ind][1]:mappings[pl_ind][1]+patch_size] = input_full[:, :, mappings[pl_ind][0]:mappings[pl_ind][0]+patch_size, mappings[pl_ind][1]:mappings[pl_ind][1]+patch_size]
        sampled_img[:, :, mappings[pl_ind][0]:mappings[pl_ind][0]+patch_size, mappings[pl_ind][1]:mappings[pl_ind][1]+patch_size] *= mask.unsqueeze(1).unsqueeze(1).unsqueeze(1).float()
    input_org = sampled_img

    return input_org

def get_action_space():
    """
    Model the action space by dividing the image space into equal size patches.
    :return: pixel mappings, img size, patch size
    """
    img_size = 256
    patch_size = 64 # 64×64×16 = 256x256
    mappings = []
    for cl in range(0, img_size, patch_size):
        for rw in range(0, img_size, patch_size):
            mappings.append([cl, rw])

    return mappings, img_size, patch_size

def get_model(model):

    if model == 'CNN':
        agent = policy_net.ConvNet()

    elif model == 'ResNet':
        agent = policy_net.ResNet(policy_net.BasicBlock, [1, 1, 1, 1], 16)  # block, layers, num_classes

    elif model == 'ResNet18':
        agent = policy_net.resnet18(num_classes=16)

    return agent
