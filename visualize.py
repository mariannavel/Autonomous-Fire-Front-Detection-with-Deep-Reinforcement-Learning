import torch
import torch.utils.data as torchdata
from utils import utils
import matplotlib.pyplot as plt
import numpy as np

from Seg_generator import *

IMG_PATH = 'data/images/'
MSK_PATH = 'data/voting_masks'

MAX_PIXEL_VALUE = 65535 # used to normalize the image
TH_FIRE = 0.25 # fire threshold

def visualize_images(images, masks, title=''):
    # input ndarray: n x 256 x 256 x 3
    # 1, 4, 6, 9, 10, 13, 19, 21, 26, 33, 38, 42, 55, 58, 66, 67 --> fire present (train)
    # 69, 70, 78, 80, 82 -> fire present (test)
    for i, (img, msk) in enumerate(zip(images, masks)):
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.subplot(1, 2, 2)
        plt.imshow(msk)
        plt.suptitle(title)
        plt.show()

def plot_img_pipeline(I_LR, I_HR, M_LR_map, M_LR, I_HR_patch, M_HR_patch):
    """
        I_LR: low resolution image
        I_HR: high resolution image
        M_LR_map: LR image map (ground truth)
        M_LR: LR image map vectorized (?)
        I_HR_patch: selected patch in high resolution (ground truth)
        M_HR_patch: selected patch masked (ground truth)
    """

    for exp_index in range(0, 1):
        plt.subplot(2, 3, 1)
        plt.imshow(I_LR[exp_index, :, :, :])
        plt.title('LR (1st cam)')
        # plt.axis(False)

        plt.subplot(2, 3, 2)
        plt.imshow(I_HR[exp_index, :, :, :])
        plt.title('HR (ideal)')
        # plt.axis(False)

        plt.subplot(2, 3, 3)
        plt.imshow(M_LR_map[exp_index, :, :])
        plt.title('SMP MSK (1st cam)')
        # plt.axis(False)

        tt = np.argmax(M_LR[exp_index, :])

        plt.subplot(2, 3, 4)
        plt.imshow(I_HR_patch[exp_index, tt, :, :, :])
        plt.title('LR/HR (2nd cam)')
        # plt.axis(False)

        plt.subplot(2, 3, 5)

        plt.imshow(M_HR_patch[exp_index, tt, :, :])
        plt.title('Fire MSK (2nd cam)')
        # plt.axis(False)

        plt.show()


def visualize_masked_imgs(model):

    trainset, testset = utils.get_dataset(model, 'data/')
    trainloader = torchdata.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0)
    testloader = torchdata.DataLoader(testset, batch_size=32, shuffle=False, num_workers=0)
    _, _, agent = utils.get_model(model)

    agent.eval().cuda()

    mappings, img_size, patch_size = utils.action_space_model(model.split('_')[1])

    ckpt_2stream = torch.load("cv/tmp/Landsat-8/F2Stream_ckpt_E_30_train_0.98_test_0.944_R_1.32E-01")
    agent.load_state_dict(ckpt_2stream['agent'])

    matches, rewards, policies = [], [], []
    for batch_idx, (inputs, targets) in enumerate(testloader):

        if batch_idx == 0:
            print("labels:", targets)

            if (model == 'R32_Landsat-8'):
                inputs = inputs.float().permute(0, 3, 1, 2)

            # Get the low resolution images for the agent and classifier
            inputs_agent = inputs.clone().cuda()
            inputs_agent = torch.nn.functional.interpolate(inputs_agent, (16, 16))
            probs = torch.sigmoid(agent.forward(inputs_agent, model.split('_')[1], 'lr'))

            # Sample Test time Policy from Agent's Output
            policy = probs.data.clone()
            policy[policy < 0.5] = 0.0
            policy[policy >= 0.5] = 1.0

            # Get the Agent Determined Images
            # inputs = torch.nn.functional.interpolate(inputs.clone(), (64, 64))
            masked_img = utils.agent_chosen_input(inputs, policy, mappings, patch_size)
            for i in range(10):
                # plt.figure(figsize=(2.5, 1.5))
                _, ax = plt.subplots(1, 2)
                ax[0].imshow(inputs[i+5].permute(1, 2, 0))
                ax[1].imshow(masked_img[i+5].permute(1, 2, 0).cpu())
                plt.subplot(1, 2, 1)
                plt.title("input")
                plt.subplot(1, 2, 2)
                plt.title("masked input")
                plt.savefig('out' + str(i) + '.png')
                # plt.show()
            break

def visualize_image3c_with_mask():
    img_filelist = sorted(os.listdir(IMG_PATH))
    msk_filelist = sorted(os.listdir(MSK_PATH))
    for fn_img, fn_mask in zip(img_filelist, msk_filelist):

        img = os.path.join(IMG_PATH, fn_img)
        img3c = get_img_762bands(img) # 3 channels
        mask = get_mask_arr(os.path.join(MSK_PATH, fn_mask))

        plt.subplot(1, 2, 1)
        plt.imshow(img3c)
        plt.title('Original image 3c')

        plt.subplot(1, 2, 2)
        plt.imshow(mask)
        plt.title('Voting mask (target)')

        plt.show()

def visualize_with_seg_mask(img3c, mask):
    # permute for visualization purposes
    img3c = img3c.float().permute(2, 1, 0)
    mask = mask.float().permute(2, 1, 0)
    # mask = torch.unsqueeze(mask[0] > 0, dim=0) # make it binary
    mask = mask > TH_FIRE
    plt.subplot(1, 2, 1)
    plt.imshow(img3c.detach().numpy())
    plt.title('Original image 3c')

    plt.subplot(1, 2, 2)
    plt.imshow(mask.detach().numpy())
    plt.title('Voting mask (target)')

    plt.show()

def baseline_vs_agent_sampling(orig, env1, env2):
    env1 = env1.float().permute(2, 1, 0)
    env2 = env2.float().permute(2, 1, 0)
    orig = orig.float().permute(2, 1, 0)

    plt.subplot(1, 3, 1)
    plt.imshow(orig.detach().numpy())
    plt.title('Original image')

    plt.subplot(1, 3, 2)
    plt.imshow(env1.detach().numpy())
    plt.title('Baseline actions')

    plt.subplot(1, 3, 3)
    plt.imshow(env2.detach().numpy())
    plt.title('Sampled actions')

    plt.show()