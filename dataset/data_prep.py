import numpy as np
import pickle
from scipy.io import loadmat
# from sklearn.model_selection import train_test_split
import os
from utils.unet_utils import get_img_762bands, get_mask_arr

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

NUM_SAMPLES = 100
IMG_PATH = "data/images100"
VEC_TARGET_PATH = f"data/{NUM_SAMPLES}/regular_split/custom_targets/thres0.01"
MSK_TARGET_PATH = "data/voting_masks100"
SAVE_PATH = f"data/{NUM_SAMPLES}/regular_split/mask_label/"

def load_mat_data(data_dir = "data/Landsat-8/"):
    """
    Load .mat dataset samples in memory.
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

def load_images(img_path, max_num=6179):
    img_filelist = sorted(os.listdir(img_path))
    images = []
    for i, fn_img in enumerate(img_filelist):
        if i == max_num: break
        img3c = get_img_762bands(os.path.join(img_path, fn_img))
        if img3c is not None:
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

def load_dict(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def load_masks_as_dict(path, max_num):
    masks = {}
    for i, filename in enumerate(sorted(os.listdir(path))):
        if i == max_num: break
        img = get_mask_arr(os.path.join(path, filename))
        if img is not None:
            masks[filename] = img
    return masks

def make_mask_target_dset(img_path, target_path, savedir, max_num=100, test_ratio=0.15):
    """
    Saves the images and their segmentation masks (targets) as train-test set
    for training the policy network in a RL framework.
    :param img_path: the path of the actual dataset images
    :param targets_path: the path of the voting mask-targets
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

    # save them to savedir as train.pkl, test.pkl
    split_and_save(np.asarray(images), np.asarray(masks), savedir, split_ratio=test_ratio)

def make_custom_target_dset(img_path, targets_path, savedir, max_num=100, test_ratio=0.15):
    """
    Saves the images and custom targets as train-test set for training
    policy network standalone.
    :param img_path: the path of the actual dataset images
    :param targets_path: the path of the vector targets
    """
    with open(targets_path, "rb") as fp:
        targets = pickle.load(fp) # list of vectors (lists)

    images = load_images(img_path, max_num=max_num)

    split_and_save(np.asarray(images), np.asarray(targets), savedir, split_ratio=test_ratio)

def make_threshold_dsets(img_path, targets_path, savedir, test_ratio=0.15):
    """
    :param img_path: path of actual Landsat-8 images
    :param targets_path: path of binary files with custom targets
    :param savedir: path to save the .pkl datasets
    :return:
    """
    fire_thresholds = (0.01, 0.02, 0.03, 0.04, 0.05)
    for _ in fire_thresholds:

        if not os.path.exists(savedir):
            os.makedirs(savedir)

        make_custom_target_dset(img_path=img_path,
                                targets_path=targets_path,
                                savedir=savedir,
                                max_num=NUM_SAMPLES,
                                split_ratio=test_ratio)
