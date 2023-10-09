"""
This file makes the data preparation before training of the PatchDrop components.
"""

import numpy as np
import pickle
from scipy.io import loadmat
# from sklearn.model_selection import train_test_split
import os
from unet.utils import get_img_762bands, get_mask_arr

NUM_SAMPLES = 2048 # I have memory error with more than 2000 data (cannot dump)

def load_mat_data(data_dir = "data/Landsat-8/"):
    """
    Load .mat dataset of 118 samples in memory.
    :param data_dir
    """
    D = loadmat(data_dir+"AFD_data_17_classes.mat")
    # for key, value in D.items():
    #     print(key)
    I_HR=D['I_HR'] # 256 x 256 x 3
    I_LR=D['I_LR'] # 32 x 32 x 3
    M_HR=D['M_HR'] # 256 x 256 [mask of big HR image]
    M_LR_map=D['M_LR'] # 4 x 4 (action array)
    M_LR_vec=D['M_LR_vec'] # 16+1 (vectorized action array)
    # 16x64x64 == 256x256
    I_HR_patch=D['I_HR_patch'] # 17 patches of 64x64x3
    M_HR_patch=D['M_HR_patch'] # 17 patches of 64x64

    # Cut off 3 channels out of 10
    # "we used channels c7 , c6 and c2 to compose the RGB bands,
    # however, the original patches contain 10 bands"
    I_HR=I_HR[:,:,:,[6,5,1]]
    I_LR=I_LR[:,:,:,[6,5,1]]
    I_HR_patch=I_HR_patch[:,:,:,:,[6,5,1]]

    I_HR=I_HR/I_HR.max()
    I_LR=I_LR/I_LR.max()
    I_HR_patch=I_HR_patch/I_HR_patch.max()

    # plot_img_pipeline(I_LR, I_HR, M_LR_map, M_LR_vec, I_HR_patch, M_HR_patch)

def train_test_split(data, labels, test_size):
    """
    Provides a split on the data based on the given test size.
    :param data (array-like): The input data
    :param labels (array-like): The corresponding labels
    :param test_size: percentage between 0 and 1 -> size of test set
    :return: train and test data, train and test labels
    """

    num_samples = len(data)
    num_test_samples = int(np.ceil(test_size * num_samples))
    # shuffled_indices = np.random.permutation(num_samples)
    test_indices = np.arange(0, num_test_samples)
    train_indices = np.arange(num_test_samples, num_samples)

    X_train = data[train_indices]
    X_test = data[test_indices]
    y_train = labels[train_indices]
    y_test = labels[test_indices]

    return X_train, X_test, y_train, y_test

def split_and_save(data, labels, savedir="data/", split_ratio=0.2):
    """
    Split data to train-test set and save them to given directory.
    :param data: HR images
    :param savedir: path to save the split data
    """
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=split_ratio)
    trainset = {"data": X_train, "targets": y_train}
    testset = {"data": X_test, "targets": y_test}

    with open(savedir + 'train.pkl', 'wb') as f:
        pickle.dump(trainset, f)
    with open(savedir + 'test.pkl', 'wb') as f:
        pickle.dump(testset, f)

def load_masks_from_folder(folder, max_num):
    images = {}
    for i, filename in enumerate(sorted(os.listdir(folder))):
        if i == max_num: break
        img = get_mask_arr(os.path.join(folder, filename)) #cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images[filename] = img
    return images

def make_PN_SIG_dset(img_path, target_path, max_num, savedir, split_ratio):
    """
    Saves the images and their segmentation masks (targets) as train-test set
    for the training of PatchDrop with Semantic Image Segmentation.
    """

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    img_filelist = sorted(os.listdir(img_path))
    msk_filelist = sorted(os.listdir(target_path))

    images = []
    masks = []
    for i, (fn_img, fn_mask) in enumerate(zip(img_filelist, msk_filelist)):

        if i == max_num: break  # because the amount of data causes SIGKILL ... to be examined
        img3c = get_img_762bands(os.path.join(img_path, fn_img))  # give the image path
        mask = get_mask_arr(os.path.join(target_path, fn_mask)) # mask path
        images.append(img3c)
        masks.append(mask)
        print(f"saved {fn_img}")

    split_and_save(np.asarray(images), np.asarray(masks), savedir, split_ratio=split_ratio)

def make_PN_dset(img_path, targets_path, savedir, max_num, split_ratio):
    """
    Takes the images path and the PN targets' path and saves them as train-test set
    for the training of Policy Network standalone.
    :param img_path: the path of the dataset images
    :param targets_path: the path of the binary vector custom targets
    """

    images = []

    with open(targets_path, "rb") as fp:
        labels = pickle.load(fp) # list of vectors (lists)

    for i, fn_img in enumerate(sorted(os.listdir(img_path))):
        if i == max_num: break
        img3c = get_img_762bands(os.path.join(img_path, fn_img))  # give the image path
        # label = labels[fn_img.replace('RT_', 'RT_Voting_')]
        images.append(img3c)
        # print(f"saved {fn_img}")

    split_and_save(np.asarray(images), np.asarray(labels), savedir, split_ratio=split_ratio)


if __name__ == "__main__":
# seg_masks = load_masks_from_folder("data/voting_masks6179", max_num=NUM_SAMPLES)

    make_PN_SIG_dset(img_path="data/images6179",
             target_path=f"data/voting_masks6179",
             max_num=NUM_SAMPLES,
             savedir= f"data/{NUM_SAMPLES}/mask_labels/rand_sampled/",
             split_ratio=0.15)

# fire_thresholds = (0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2)
# for thres in fire_thresholds:
#
#     savedir = f"pretrainPN/threshold_experiment/{NUM_SAMPLES}/thres{thres}/data/"
#     if not os.path.exists(savedir):
#         os.makedirs(savedir)
#
#     make_PN_dset(img_path="data/images6179",
#              targets_path=f"pretrainPN/threshold_experiment/{NUM_SAMPLES}/custom_targets/thres{thres}",
#              savedir=savedir,
#              max_num=NUM_SAMPLES,
#              split_ratio=0.15)