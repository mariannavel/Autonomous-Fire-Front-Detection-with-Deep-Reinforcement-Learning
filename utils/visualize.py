import matplotlib.pyplot as plt
import numpy as np

MAX_PIXEL_VALUE = 65535  # used to normalize the image
TH_FIRE = 0.25  # fire threshold

def visualize_images(set, title=''):
    # input ndarray: 256 x 256 x 3
    for image in set:
        plt.imshow(image)
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

def visualize_image_pairs(img1, img2, title="", savepath=""):
    # input ndarray: n x 256 x 256 x 3
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

def visualize_img3c_mask(img3c, mask):
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

def plot_inference_grid(HR_img, masked_HR_img, target):
    # plt.figure(figsize=(16, 10))
    plt.subplots(1, 3, figsize=(9, 3))

    plt.subplot(1, 3, 1)
    plt.imshow(HR_img)
    # plt.title("HR image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(masked_HR_img)
    # plt.title("Masked HR image")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(target)
    # plt.title("Segmentation mask")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def plot_pipeline(I_LR, I_HR, M_LR_map, M_LR, I_HR_patch, M_HR_patch):
    """
    Plot image of initial .mat dataset through the pipeline.
        I_LR: low resolution image
        I_HR: high resolution image
        M_LR_map: LR image map (ground truth)
        M_LR: LR image map vectorized
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

def plot_baseline_vs_agent_policy(image, env1, env2):
    """
    Plot original image with 1. masked resulting from baseline policy
                             2. masked resulting from agent policy
    :param image: [tensor of 1 x dim x dim]
    :param env1: [tensor of 1 x dim x dim] the baseline
    :param env2: [tensor of 1 x dim x dim] the agent
    :return:
    """
    image = image.float().permute(2, 1, 0)
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
