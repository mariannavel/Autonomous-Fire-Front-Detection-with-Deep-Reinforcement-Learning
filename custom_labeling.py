import numpy as np
from utils.agent_utils import get_action_space
import pickle
from data_prep import load_masks_as_dict
import os

NUM_SAMPLES = 100

def count_fire_pixels(images):
    """
    images: list of binary images (ndarray)
    :return: list of number of fire pixels of each image
    """
    img_fire_pixels = []
    for bin_arr in images:
        unique, counts = np.unique(bin_arr, return_counts=True)
        # print(dict(zip(unique, counts)))
        # counts[0]: how many zeros
        # counts[1]: how many ones
        if len(counts) > 1: # has fire pixels
            img_fire_pixels.append(counts[1])
        else:
            img_fire_pixels.append(0)

    return img_fire_pixels

def split_in_patches(mask, N=16):
    """
    Splits the given image in N patches.
    :return: [list] N patches of equal size
    """
    mappings, patch_size = get_action_space()
    patches = []
    for i in range(N):
        buf = mappings[i]
        patch = mask[buf[0]:buf[0]+patch_size, buf[1]:buf[1]+patch_size]
        patches.append(patch)

    return patches

def get_label(num_fire_pixels, patch_size=64, fire_thres=0.05):
    """
    Returns the multi-label ground truth of an image based on the fire threshold.

    num_fire_pixels: [list] the number of fire pixels of each patch in an image
    :return: [ndarray] a vector of 1 where fire pixels are over threshold
             or 0 where they are under threshold
    """
    label = np.zeros(16)
    for i, num in enumerate(num_fire_pixels):
        fire_percentage = num/patch_size # * 100
        if fire_percentage > fire_thres:
            label[i] = 1
        else:
            label[i] = 0
    return label

def make_custom_labels(seg_masks, fire_thres, savepath=f"data/custom_targets"):
    """
    This function returns the labels of Policy Network as binary vectors,
    based on the percentage of fire pixels in each patch.

    :param seg_masks: the binary segmentation masks
    """
    label_vec = []
    for mask_key in seg_masks:
        patches = split_in_patches(seg_masks[mask_key]) # list of 16 patches
        num_fire_pixels = count_fire_pixels(patches)
        label_vec.append(get_label(num_fire_pixels, fire_thres=fire_thres))

    with open(savepath+f"thres{fire_thres}", "wb") as fp:
        pickle.dump(label_vec, fp)
    # return label_vec

if __name__ == "__main__":

    seg_masks = load_masks_as_dict("data/voting_masks100", max_num=NUM_SAMPLES)
    fire_thresholds = (0.01, 0.02, 0.03, 0.04, 0.05)
    for thres in fire_thresholds:
        savepath = f"pretrainResNet/{NUM_SAMPLES}/custom_targets/"
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        make_custom_labels(seg_masks, thres, savepath=savepath)
