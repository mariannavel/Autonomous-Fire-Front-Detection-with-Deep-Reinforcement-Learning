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
import matplotlib.pyplot as plt
from unet.utils import get_img_762bands, get_mask_arr
from custom_labeling import split_in_patches, count_fire_pixels, get_label
from utils.visualize import visualize_image, visualize_images
from data_prep import train_test_split
import pickle
import random

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

random.seed(42)

NUM_SAMPLES = 4096

def load_data_dict(img_path, msk_path, max_num=6179):
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

def load_images(img_path, max_num=6179):
    img_filelist = sorted(os.listdir(img_path))
    images = []
    for i, fn_img in enumerate(img_filelist):
        if i == max_num: break
        img3c = get_img_762bands(os.path.join(img_path, fn_img))
        images.append(img3c)
    return images

def load_masks(msk_path, max_num=6179):
    msk_filelist = sorted(os.listdir(msk_path))
    masks = []
    for i, fn_mask in enumerate(msk_filelist):
        if i == max_num: break
        mask = get_mask_arr(os.path.join(msk_path, fn_mask))
        masks.append(mask)
    return masks

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

def discriminate_fire_present(bin_labels):
    """
    :param bin_labels: list of binary vector labels
    :return: f_img_tuples : <image index, num of fire patches> per fire image
            nf_img_idx : <image index> per non-fire image
    """

    f_img_tuples, nf_img_idx = [], []

    # save indexes of fire / non-fire images
    for i, label in enumerate(bin_labels):
        num_fire_patches = sum(label)
        if num_fire_patches == 0:
            nf_img_idx.append(i) # it is a non-fire index
        else:
            f_img_tuples.append((i, num_fire_patches)) # fire index

    return f_img_tuples, nf_img_idx

def train_test_fire_split(fire_set, non_fire_set, f_prior, nf_prior, num_test):
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

def get_fire_patch_occurr(fire_img_tuples):
    """
    Returns a dictionary where *keys* are fire patch counts and *values* are
    the number of images having key number of fire patches.
    """
    num_fire = [int(tup[1]) for tup in fire_img_tuples]
    counts_set = set(num_fire) # unique fire patch counts
    dict = {}
    for f_count in counts_set:
        print(f"{f_count} fire patches: {num_fire.count(f_count)} occurrences")
        dict[str(f_count)] = num_fire.count(f_count)
    return dict

def get_fire_patch_count_sets(fire_img_tuple, images, labels):
    """
    Returns a dictionary where *keys* are fire patch counts and *values* are
    the actual images having the corresponding number of fire patches.
    """
    fire_indexes, num_fire = [], []
    for tup in fire_img_tuple:
        fire_indexes.append(tup[0])
        num_fire.append(int(tup[1]))

    counts_set = set(num_fire)
    dict_images, dict_labels = {}, {}
    for f_count in counts_set:
        dict_images[str(f_count)] = [] # in this list will save the images
        dict_labels[str(f_count)] = []

    # Add images to the corresponding set
    for i in fire_indexes:
        key = num_fire[i]
        dict_images[str(key)].append(images[ fire_indexes[i] ]) # --> the fire image
        dict_labels[str(key)].append(labels[ fire_indexes[i] ])

    return dict_images, dict_labels

def save_dataset(savedir, trainset, testset):
    with open(savedir + 'train.pkl', 'wb') as f:
        pickle.dump(trainset, f)
    with open(savedir + 'test.pkl', 'wb') as f:
        pickle.dump(testset, f)

def balanced_split(fire_img_tuple, non_fire_img_idx, targets, test_ratio):
    """
    :param fire_img_tuple : <image index, num of fire patches> per fire image
    :param non_fire_img_idx : <image index> per non-fire image
    :param test_ratio: percentage of test set
    :return: train_set, test_set with proportionally similar label distributions
    """
    images = np.array(data["images"])

    f_count = len(fire_img_tuple)
    nf_count = len(non_fire_img_idx)
    f_prior = f_count / len(images)
    nf_prior = nf_count / len(images)

    print("Fire-present images: %.3f \tNon-fire images: %.3f" % (f_prior, nf_prior))

    # Randomize fire images set
    random.shuffle(fire_img_tuple)

    # Sample test_ratio for testing and rest for training

    num_test = test_ratio * len(images)
    # num_train = len(images) - num_test
    findex = [tup[0] for tup in fire_img_tuple] # separate indexes from fire patch count

    X_train, X_test = train_test_fire_split(images[findex], images[non_fire_img_idx], f_prior, nf_prior, num_test)
    y_train, y_test = train_test_fire_split(targets[findex], targets[non_fire_img_idx], f_prior, nf_prior, num_test)

    trainset = {"data": X_train, "targets": y_train}
    testset = {"data": X_test, "targets": y_test}

    return trainset, testset

def randomize_dataset(X_train, y_train, X_test, y_test):
    # Randomize training data
    shuffled_idx = [i for i in range(len(X_train))]
    random.shuffle(shuffled_idx)
    X_train = [X_train[i] for i in shuffled_idx]
    y_train = [y_train[i] for i in shuffled_idx]
    # Randomize test data
    shuffled_idx = [i for i in range(len(X_test))]
    random.shuffle(shuffled_idx)
    X_test = [X_test[i] for i in shuffled_idx]
    y_test = [y_test[i] for i in shuffled_idx]

    return X_train, X_test, y_train, y_test

def stratify_multi_split(dict_images, dict_labels, test_ratio):
    """
    :param dict_images --> *keys* are num fire patches, *values* are the images
    :return: trainset, testset --> stratified, randomized
    """
    X_train_all, y_train_all = [], []
    X_test_all, y_test_all = [], []
    for key in dict_images:
        images = np.array(dict_images[key])
        labels = np.array(dict_labels[key])
        # Split each set to train and test
        X_train, X_test, y_train, y_test = train_test_split(images, labels, test_ratio)
        X_train_all.extend(X_train)
        y_train_all.extend(y_train)
        X_test_all.extend(X_test)
        y_test_all.extend(y_test)

    # Randomize sets; otherwise patch keys will be in order
    X_train, X_test, y_train, y_test = randomize_dataset(X_train_all, y_train_all,
                                                         X_test_all, y_test_all)

    trainset = {"data": np.array(X_train), "targets": np.array(y_train)}
    testset = {"data": np.array(X_test), "targets": np.array(y_test)}
    return trainset, testset

def dset_make_randomized():
    data = load_data_dict(img_path="data/images6179", msk_path="data/voting_masks6179")

    data["bin_vec_labels"] = get_custom_labels(data["masks"], fire_thres=0.01)

    fire_img_tuples, non_fire_img_idx = discriminate_fire_present(bin_vec_labels)

    # masks = np.array(data["masks"])
    bin_labels = np.array(data["bin_vec_labels"])

    trainset, testset = balanced_split(fire_img_tuples, non_fire_img_idx, targets=bin_labels, test_ratio=0.20)

    save_dataset(f"data/balanced_splits/{NUM_SAMPLES}/", trainset, testset)

def barplot(num_dict):
    fire_patch_count = list(num_dict.keys())
    num_images = list(num_dict.values())

    # fig = plt.figure(figsize=(10, 5))
    plt.bar(fire_patch_count, num_images, color='blue', width=0.5) # width=0.4
    plt.xlabel("fire patches")
    plt.ylabel("images")
    # plt.title("Distribution of fire patches")
    plt.show()

if __name__ == "__main__":

    masks = load_masks(msk_path="data/voting_masks6179", max_num=NUM_SAMPLES)
    images = load_images(img_path="data/images6179", max_num=NUM_SAMPLES)

    bin_vec_labels = get_custom_labels(masks, fire_thres=0.01)

    fire_img_tuples, non_fire_img_idx = discriminate_fire_present(bin_vec_labels)
    # --> Landsat-8 Europe dataset has only fire-present images !

    num_dict = get_fire_patch_occurr(fire_img_tuples)

    barplot(num_dict)

    # Get the image set of each fire patch count
    # dict_images, dict_labels = get_fire_patch_count_sets(fire_img_tuples, images, masks)
    #
    # trainset, testset = stratify_multi_split(dict_images, dict_labels, test_ratio=0.15)
    #
    # save_dataset(f"data/{NUM_SAMPLES}/mask_labels/stratified/", trainset, testset)
