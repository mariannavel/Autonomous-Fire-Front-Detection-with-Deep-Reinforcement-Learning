import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as torchdata
import torchvision.models as torchmodels
import numpy as np
import shutil
# from random import randint, sample
from sklearn.model_selection import train_test_split
from utils.custom_dataloader import CustomDatasetFromImages, LandsatDataset, LandsatSubset


def save_args(__file__, args):
    shutil.copy(os.path.basename(__file__), args.cv_dir)
    with open(args.cv_dir+'/args.txt','w') as f:
        f.write(str(args))

def performance_stats(policies, rewards, dice_coef):
    # Print the performace metrics including the average reward, average number
    # and variance of sampled num_patches, and number of unique policies
    policies = torch.cat(policies, 0)
    # rewards = torch.tensor(rewards)
    rewards = torch.cat(rewards, 0)
    # accuracy = torch.cat(matches, 0).mean()

    reward = rewards.mean()
    dice_coef = sum(dice_coef)/len(dice_coef)
    num_unique_policy = policies.sum(1).mean()
    variance = policies.sum(1).std()
    # posa einai ta policies ?? prepei oso to batch size!
    policy_set = [p.cpu().numpy().astype(np.int).astype(np.str) for p in policies]
    policy_set = set([''.join(p) for p in policy_set])

    return reward, num_unique_policy, variance, policy_set, dice_coef

def compute_reward(preds, targets, policy, penalty):
    # Reward function favors policies that drops patches only if the classifier
    # successfully categorizes the image
    patch_use = policy.sum(1).float() / policy.size(1)
    sparse_reward = 1.0 - patch_use**2

    _, pred_idx = preds.max(1)
    match = (pred_idx==targets).data

    reward = sparse_reward
    reward[~match] = penalty
    reward = reward.unsqueeze(1)

    return reward, match.float()

def get_transforms(rnet, dset):

    if dset=='C10' or dset=='C100':
        mean = [x/255.0 for x in [125.3, 123.0, 113.9]]
        std = [x/255.0 for x in [63.0, 62.1, 66.7]]
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ])

    elif dset=='ImgNet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(32), #224
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ])

        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(32), #224
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ])

    elif dset=='fMoW':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform_train = transforms.Compose([
           transforms.resise(224),
           transforms.RandomCrop(224),
           transforms.ToTensor(),
           transforms.Normalize(mean, std)
        ])
        transform_test = transforms.Compose([
           transforms.resize(224),
           transforms.CenterCrop(224),
           transforms.ToTensor(),
           transforms.Normalize(mean, std)
        ])

    return transform_train, transform_test

def agent_chosen_input(input_org, policy, mappings, patch_size):
    """ Generate masked images w.r.t policy learned by the agent.
    """
    input_full = input_org.clone()
    sampled_img = torch.zeros([input_org.shape[0], input_org.shape[1], input_org.shape[2], input_org.shape[3]])
    for pl_ind in range(policy.shape[1]):
        mask = (policy[:, pl_ind] == 1).cpu()
        sampled_img[:, :, mappings[pl_ind][0]:mappings[pl_ind][0]+patch_size, mappings[pl_ind][1]:mappings[pl_ind][1]+patch_size] = input_full[:, :, mappings[pl_ind][0]:mappings[pl_ind][0]+patch_size, mappings[pl_ind][1]:mappings[pl_ind][1]+patch_size]
        sampled_img[:, :, mappings[pl_ind][0]:mappings[pl_ind][0]+patch_size, mappings[pl_ind][1]:mappings[pl_ind][1]+patch_size] *= mask.unsqueeze(1).unsqueeze(1).unsqueeze(1).float()
    input_org = sampled_img

    return input_org.cuda()

def action_space_model(dset):
    # Model the action space by dividing the image space into equal size patches
    if dset == 'C10' or dset == 'C100':
        img_size = 32
        patch_size = 8
    elif dset == 'fMoW':
        img_size = 224
        patch_size = 56
    elif dset == 'ImgNet':
        img_size = 224
        patch_size = 56
    elif dset == 'Landsat-8':
        img_size = 256
        patch_size = 64 # 64×64×16 = 256x256

    mappings = []
    for cl in range(0, img_size, patch_size):
        for rw in range(0, img_size, patch_size):
            mappings.append([cl, rw])

    return mappings, img_size, patch_size

# Pick from the datasets available and the hundreds of models we have lying around depending on the requirements.
def get_dataset(model, root='data/'):
    
    if model=='R_Landsat8':
        trainset = LandsatDataset(root+'train85.pkl')
        testset = LandsatDataset(root+'test15.pkl')
    else:
        rnet, dset = model.split('_')
        transform_train, transform_test = get_transforms(rnet, dset) # edw mporw na kanw resize tis eikones

        if dset=='C10':
            trainset = torchdata.CIFAR10(root=root, train=True, download=True, transform=transform_train)
            testset = torchdata.CIFAR10(root=root, train=False, download=True, transform=transform_test)
        elif dset=='C100':
            trainset = torchdata.CIFAR100(root=root, train=True, download=True, transform=transform_train)
            testset = torchdata.CIFAR100(root=root, train=False, download=True, transform=transform_test)
        elif dset=='ImgNet':
            trainset = torchdata.ImageFolder(root+'/ImageNet/train/', transform_train)
            testset = torchdata.ImageFolder(root+'/ImageNet/test/', transform_test)
        elif dset=='fMoW':
            trainset = CustomDatasetFromImages(root+'/fMoW/train.csv', transform_train)
            testset = CustomDatasetFromImages(root+'/fMoW/test.csv', transform_test)
    
    # print(type(trainset))
    return trainset, testset

def get_model(model):

    from models import resnet_cifar, resnet_in

    if model=='R32_C10':
        # [3,4,6,3] -> 3 blocks sto 1o layer, 4 blocks sto 2o, k.o.k
        # 3 -> kernel size pou tha efarmostei sto arxiko input
        # 10 -> number of classes
        # Ta ResNet HR kai LR kanoun classify ta antikeimena stis eikones
        rnet_hr = resnet_cifar.ResNet(resnet_cifar.BasicBlock, [3,4,6,3], 3, 10)
        rnet_lr = resnet_cifar.ResNet(resnet_cifar.BasicBlock, [3,4,6,3], 3, 10)
        # O Agent kanei sample x out of 16 patches
        agent = resnet_cifar.ResNet(resnet_cifar.BasicBlock, [1,1,1,1], 3, 16)

    elif model=='R32_C100':
        rnet_hr = resnet_cifar.ResNet(resnet_cifar.BasicBlock, [3,4,6,3], 3, 100)
        rnet_lr = resnet_cifar.ResNet(resnet_cifar.BasicBlock, [3,4,6,3], 3, 100)
        agent = resnet_cifar.ResNet(resnet_cifar.BasicBlock, [1,1,1,1], 3, 16)

    elif model=='R50_ImgNet':
        rnet_hr = resnet_in.ResNet(resnet_in.BasicBlock, [3,4,6,3], 7, 1000)
        rnet_lr = resnet_in.ResNet(resnet_in.BasicBlock, [3,4,6,3], 7, 1000)
        agent = resnet_in.ResNet(resnet_in.BasicBlock, [2,2,2,2], 3, 16)

    elif model=='R34_fMoW':
        rnet_hr = resnet_in.ResNet(resnet_in.BasicBlock, [3,4,6,3], 7, 62)
        rnet_lr = resnet_in.ResNet(resnet_in.BasicBlock, [3,4,6,3], 7, 62)
        agent = resnet_in.ResNet(resnet_in.BasicBlock, [2,2,2,2], 3, 16)

    elif model=='Landsat8_ResNet':
        # rnet_hr = resnet_in.ResNet(resnet_in.BasicBlock, [3,4,6,3], 2)
        # rnet_lr = resnet_in.ResNet(resnet_in.BasicBlock, [3,4,6,3], 2)
        agent = resnet_in.ResNet(resnet_in.BasicBlock, [1,1,1,1], 16) # block, layers, num_classes

    return agent # rnet_hr, rnet_lr,

# def keras2pytorch_model(keras_model, pytorch_model):

