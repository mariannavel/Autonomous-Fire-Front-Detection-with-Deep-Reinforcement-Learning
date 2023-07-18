import numpy as np
import pickle
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

IMG_PATH = 'data/images6179/'
MSK_PATH = 'data/voting_masks6179'

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
    Load .mat dataset of 118 samples.
    :param data_dir:
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