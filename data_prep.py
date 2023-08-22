"""
This file makes the data preparation before training of the PatchDrop components.
"""

import numpy as np
import pickle
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from SegNet.utils import get_img_762bands, get_mask_arr
from visualize import visualize_image
from EDA import split_in_patches, count_fire_pixels, get_label

IMG_PATH = 'data/images100/'
MSK_PATH = 'data/voting_masks100'

def binarize_masks(M_LR_vec):
    """
    Called when training PatchDrop on Landsat-8 with classifier.
    :param M_LR_vec: 118 x 17 ndarray
    :return: boolean vector with value True corresponding to images with fire
    """
    # the last column indicates whether there is no patch to sample
    # 1 for entries that have no fire
    last_col = M_LR_vec[:,-1]
    # I need to take the complement of the last column
    return [not val for val in last_col]

def save_image(img, path):
    plt.imshow(img)
    plt.axis(False)
    plt.savefig(path)

def split_and_save(data, labels, data_dir = "data/"):
    """
    Split data to train-test set and save them to given directory.
    :param data: HR images
    :param data_dir: path to save the split data
    """
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.15)
    trainset = {"data": X_train, "targets": y_train}
    testset = {"data": X_test, "targets": y_test}

    with open(data_dir + 'train.pkl', 'wb') as f:
        pickle.dump(trainset, f)
    with open(data_dir + 'test.pkl', 'wb') as f:
        pickle.dump(testset, f)

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

# bool_targets = binarize_masks(M_LR_vec) # 118 entries
# for i, target in enumerate(bool_targets):
#     if target == True:
#         save_image(I_HR[i], "HR_image"+str(i)+".png")
#         save_image(I_LR[i], "LR_image"+str(i)+".png")

def load_images_from_folder(folder):
    images = {}
    for filename in sorted(os.listdir(folder)):
        img = get_mask_arr(os.path.join(folder, filename)) #cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images[filename] = img
    return images

def make_img_labels(seg_masks):
    """
    This function returns the labels of Policy Network as binary vectors,
    based on the percentage of fire pixels in each patch.

    :param seg_masks: the binary segmentation masks (returned by EDA.load_images_from_folder("data/voting_masks100"))
    """
    label_vec = []
    for mask_key in seg_masks:
        patches = split_in_patches(seg_masks[mask_key]) # list of 16 patches
        num_fire_pixels = count_fire_pixels(patches)
        label_vec.append(get_label(num_fire_pixels))

    # with open("data/agent_targets", "wb") as fp:
    #     pickle.dump(label_vec, fp)
    return label_vec

def make_PN_SIG_dset():
    """
    Saves the images and their segmentation masks (targets) as train-test set
    for the training of PatchDrop with Semantic Image Segmentation.
    """
    img_filelist = sorted(os.listdir(IMG_PATH))
    msk_filelist = sorted(os.listdir(MSK_PATH))

    images = []
    masks = []
    for i, (fn_img, fn_mask) in enumerate(zip(img_filelist, msk_filelist)):

        if i == 4000: break  # because the amount of data causes SIGKILL ... to be examined
        img3c = get_img_762bands(os.path.join(IMG_PATH, fn_img))  # give the image path
        mask = get_mask_arr(os.path.join(MSK_PATH, fn_mask)) # mask path
        images.append(img3c)
        masks.append(mask)
        print(f"saved {fn_img}")

    split_and_save(np.asarray(images), np.asarray(masks))

def make_PN_dset(img_path=IMG_PATH, target_path="data/agent_targets"):
    """
    Takes the images path and the PN targets' path and saves them as train-test set
    for the training of Policy Network standalone.
    :param img_path: the path of the dataset images
    :param target_path: the path of the binary vector targets of PN
    """

    images = []

    with open(target_path, "rb") as fp:
        labels = pickle.load(fp) # list of 100 vectors (lists)

    for fn_img in sorted(os.listdir(img_path)):

        img3c = get_img_762bands(os.path.join(IMG_PATH, fn_img))  # give the image path
        # label = labels[fn_img.replace('RT_', 'RT_Voting_')]
        # visualize_image(img3c)
        images.append(img3c)
        print(f"saved {fn_img}")

    split_and_save(np.asarray(images), np.asarray(labels))


masks = load_images_from_folder("data/voting_masks100")
labels = make_img_labels(masks)
with open("data/agent_targets", "wb") as fp:
    pickle.dump(labels, fp)

make_PN_dset()