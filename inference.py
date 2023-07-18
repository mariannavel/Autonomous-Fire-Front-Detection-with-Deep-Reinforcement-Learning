import torch
from utils import utils
from utils.custom_dataloader import LandsatDataset
import numpy as np
from visualize import visualize_images

LR_size = 32
num_test = 21

def inference(images, mappings, patch_size):
    """ Get the agent masked images from input based
        on the policy distribution output """

    images = images.permute(0, 3, 1, 2)
    # plot the image to validate correctness...

    # Get the low resolution agent images
    input_agent = images.clone()
    input_agent = torch.nn.functional.interpolate(input_agent, (LR_size, LR_size))

    # get the probability distribution output of PN
    # probs = torch.sigmoid(PolicyNetwork(input_agent))
    probs = PolicyNetwork(input_agent)

    # Sample the test-time policy
    policy = probs.data.clone()
    policy[policy < 0.5] = 0.0
    policy[policy >= 0.5] = 1.0

    # Get the masked high-res image and perform inference
    masked_image = utils.get_agent_masked_image(images, policy, mappings, patch_size)

    return masked_image

def load_images(num, datapath):
    """ num: in [1, 21] number of images from the selected Landsat-8 dataset
        return: num tensor images """

    fire_idx = [1, 4, 6, 9, 10, 13, 19, 21, 26, 33, 38, 42, 55, 58, 66, 67, 69, 70, 78, 80, 82]

    # ndarray: n x 256 x 256 x 3
    dataset = LandsatDataset(datapath)

    # get num random images from "fired images" set
    # indexes = np.random.randint(0, len(fire_idx), num)
    indexes = np.arange(0, len(fire_idx))

    ret_data = torch.empty((num, 256, 256, 3))
    ret_masks = torch.empty((num, 256, 256, 1))
    for k, i in enumerate(indexes):
        ret_data[k] = torch.from_numpy(dataset.data[fire_idx[i]])
        ret_masks[k] = torch.from_numpy(dataset.targets[fire_idx[i]])

    return ret_data, ret_masks


# def __main__():

test_images, seg_masks = load_images(num_test, 'data/train85.pkl')

PolicyNetwork = utils.get_model('ResNet_Landsat8')
state_dict = torch.load(f"checkpoints/Policy_ckpt_E_1000_R_0.478_Res")
PolicyNetwork.load_state_dict(state_dict["agent"])
print("Loaded the trained agent!")

# get agent action space
mappings, _, patch_size = utils.action_space_model('Landsat8')

masked_images = inference(test_images, mappings, patch_size)

# transform before visualization
masked_images = masked_images.permute(0, 2, 3, 1).cpu().numpy()
seg_masks = seg_masks.cpu().numpy()
visualize_images(masked_images, seg_masks, "agent prediction vs U-Net target")