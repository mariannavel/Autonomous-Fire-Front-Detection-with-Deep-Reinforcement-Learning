"""
Explore dataset to balance:
    1. fire present images vs. non fire
    2. number of fire patches in fire present images of each set (train/test)

oi non-fire images prepei na einai perissoteres kai afto einai fysiologiko (as min ksepernoun to 50%...)
ta non-fire patches prepei na einai perissotera kai afto einai fysiologiko

count fire patches of fire-present (prepei na exoun >= 1) --> calc. mean, std ---> shuffle data before splitting to train-test (randomization)

"""
import os
import numpy as np
from unet.utils import get_img_762bands, get_mask_arr
from custom_labeling import split_in_patches, count_fire_pixels, get_label
from utils.visualize import visualize_image
import pickle
import random

random.seed(42)

NUM_SAMPLES = 1000

def load_data(img_path, msk_path, max_num=2000):
    """
    :param img_path: the images path
    :param msk_path: the segmentation masks path
    :param max_num: max number of samples to load
    :return: data_dict: dictionary of 3 lists sorted by name (filenames, images, masks)
    """

    img_filelist = sorted(os.listdir(img_path))
    msk_filelist = sorted(os.listdir(msk_path))

    data_dict = {"names": [], "images": [], "masks": []} # store them as key-value pairs

    for i, (fn_img, fn_mask) in enumerate(zip(img_filelist, msk_filelist)):

        if i == max_num: break

        img3c = get_img_762bands(os.path.join(img_path, fn_img))
        mask = get_mask_arr(os.path.join(msk_path, fn_mask))

        data_dict["names"].append(fn_img) # filenames are unique
        data_dict["images"].append(img3c)
        data_dict["masks"].append(mask)

    return data_dict

def get_custom_labels(seg_masks, fire_thres=0.05):
    """
    This function returns the labels of Policy Network as binary vectors,
    based on the percentage of fire pixels in each patch.

    :param seg_masks: the binary segmentation masks
    """
    label_vec = []
    for mask in seg_masks:
        patches = split_in_patches(mask) # list of 16 patches
        num_fire_pixels = count_fire_pixels(patches)
        label_vec.append(get_label(num_fire_pixels, fire_thres=fire_thres))

    return label_vec

def explore_fire_patches(bin_labels):
    """
    :param bin_labels: list of binary vector labels
    :return: f_img_set : <image index, num of fire patches> per fire image
            nf_img_set : <image index> per non-fire image
    """

    f_img_set, nf_img_set = [], []

    # save indexes of fire / non-fire images
    for i, label in enumerate(bin_labels):
        num_fire_patches = sum(label)
        if num_fire_patches == 0:
            nf_img_set.append(i) # it is a non-fire index
        else:
            f_img_set.append((i, num_fire_patches)) # fire index

    return f_img_set, nf_img_set

def train_test_split(fire_set, non_fire_set, f_prior, nf_prior, num_test):
    """
    Perform train-test split for a single array-like (samples or labels)
    considering fire distribution in the samples.
    :param fire_set: part of the array containing fire samples/labels
    :param non_fire_set: part containing non-fire samples/labels
    :param num_test: number of test images/labels
    :return: train-test split of array-like data
    """
    # Both sets must have both fire and non-fire images equally
    num_fire_test = int(np.round(f_prior * num_test))
    num_non_fire_test = int(np.round(nf_prior * num_test))

    test_fire = fire_set[0:num_fire_test]
    train_fire = fire_set[num_fire_test:]  # the rest for training
    test_non_fire = non_fire_set[0:num_non_fire_test]
    train_non_fire = non_fire_set[num_non_fire_test:]

    train = np.append(train_fire, train_non_fire, 0)
    test = np.append(test_fire, test_non_fire, 0)

    return train, test

def save_dataset(savedir, trainset, testset):
    with open(savedir + 'train.pkl', 'wb') as f:
        pickle.dump(trainset, f)
    with open(savedir + 'test.pkl', 'wb') as f:
        pickle.dump(testset, f)

def balanced_split(fire_img_idx, non_fire_img_idx, targets, test_ratio):
    """
    :param fire_img_idx : <image index, num of fire patches> per fire image
    :param non_fire_img_idx : <image index> per non-fire image
    :param test_ratio: percentage of test set
    :return: train_set, test_set with proportionally similar label distributions
    """
    images = np.array(data["images"])

    f_count = len(fire_img_idx)
    nf_count = len(non_fire_img_idx)
    f_prior = f_count / len(images)
    nf_prior = nf_count / len(images)

    print("Fire-present images: %.3f \tNon-fire images: %.3f" % (f_prior, nf_prior))

    # Randomize fire images set
    random.shuffle(fire_img_idx)

    # Sample test_ratio for testing and rest for training

    num_test = test_ratio * len(images)
    # num_train = len(images) - num_test
    findex = [tup[0] for tup in fire_img_idx] # separate indexes from fire patch count

    X_train, X_test = train_test_split(images[findex], images[non_fire_img_idx], f_prior, nf_prior, num_test)
    y_train, y_test = train_test_split(targets[findex], targets[non_fire_img_idx], f_prior, nf_prior, num_test)

    trainset = {"data": X_train, "targets": y_train}
    testset = {"data": X_test, "targets": y_test}

    return trainset, testset


if __name__ == "__main__":

    data = load_data(img_path="data/images6179", msk_path="data/voting_masks6179")

    data["bin_vec_labels"] = get_custom_labels(data["masks"], fire_thres=0.01)

    fire_img_set, non_fire_img_set = explore_fire_patches(data["bin_vec_labels"])

    masks = np.array(data["masks"])
    bin_labels = np.array(data["bin_vec_labels"])

    trainset, testset = balanced_split(fire_img_set, non_fire_img_set, targets=masks, test_ratio=0.15)

    save_dataset(f"data/balanced_splits/{NUM_SAMPLES}/", trainset, testset)
