import matplotlib.pyplot as plt
import numpy as np
from unet.utils import get_mask_arr, get_img_762bands
import os

MAX_PIXEL_VALUE = 65535 # used to normalize the image
TH_FIRE = 0.25 # fire threshold

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
    # 1, 4, 6, 9, 10, 13, 19, 21, 26, 33, 38, 42, 55, 58, 66, 67 --> fire present (train)
    # 69, 70, 78, 80, 82 -> fire present (test)
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
        # plt.savefig(f"{savepath}myplot{i}")

def plot_img_pipeline(I_LR, I_HR, M_LR_map, M_LR, I_HR_patch, M_HR_patch):
    """
    Plot image of initial .mat dataset through the pipeline.
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

def plot_images3c_with_mask(img_path='data/images/', msk_path='data/voting_masks'):

    img_filelist = sorted(os.listdir(img_path))
    msk_filelist = sorted(os.listdir(msk_path))
    for fn_img, fn_mask in zip(img_filelist, msk_filelist):

        img = os.path.join(img_path, fn_img)
        img3c = get_img_762bands(img) # 3 channels
        mask = get_mask_arr(os.path.join(msk_path, fn_mask))

        plt.subplot(1, 2, 1)
        plt.imshow(img3c)
        plt.title('Original image 3c')

        plt.subplot(1, 2, 2)
        plt.imshow(mask)
        plt.title('Voting mask (target)')

        plt.show()

def plot_baseline_vs_agent(image, env1, env2):
    """
    Plot original image with 1. masked resulting from baseline policy
                             2. masked resulting from agent policy
    :param image: [tensor of 1 x dim x dim]
    :param env1: [tensor of 1 x dim x dim] the baseline
    :param env2: [tensor of 1 x dim x dim] the agent
    :return:
    """
    image = image.float().permute(2, 1, 0) # 2, 1 ??
    env1 = env1.float().permute(2, 1, 0)
    env2 = env2.float().permute(2, 1, 0)

    plt.subplot(1, 3, 1)
    plt.imshow(image.detach().numpy())
    plt.title('Original image')

    plt.subplot(1, 3, 2)
    plt.imshow(env1.detach().numpy())
    plt.title('Baseline actions')

    plt.subplot(1, 3, 3)
    plt.imshow(env2.detach().numpy())
    plt.title('Sampled actions')

    plt.show()