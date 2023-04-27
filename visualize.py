import torch
import torch.utils.data as torchdata
from utils import utils
import matplotlib.pyplot as plt
import numpy as np

def save_image(img, name):
    plt.imshow(img)
    plt.axis(False)
    plt.savefig(name)

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

    # ckpt_hr_and_agent = "cv/tmp/ckpt_E_10_A_0.866_R_-1.03"
    # ckpt_lr = torch.load("cv/tmp/LR_ckpt_E_5_A_0.551")
    # rnet_lr.load_state_dict(ckpt_lr['state_dict'])
    # checkpoint = torch.load(ckpt_hr_and_agent)
    # rnet_hr.load_state_dict(checkpoint['resnet_hr'])
    # agent.load_state_dict(checkpoint['agent'])
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

# visualize_masked_imgs("R32_Landsat-8")