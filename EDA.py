"""
This file performs Exploratory Data Analysis on the images of Landsat-8 dataset.
"""
import numpy as np
import cv2
import os
import rasterio
from visualize import visualize_image
import matplotlib.pyplot as plt
from utils.utils import action_space_model
import pickle

FIRE_THRES = 0.05

def get_mask_arr(path):
    img = rasterio.open(path).read().transpose((1, 2, 0))
    seg = np.float32(img)
    return seg

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
        # visualize_image(bin_arr)
        if len(counts) > 1: # has fire pixels
            img_fire_pixels.append(counts[1])
        else:
            img_fire_pixels.append(0)

    return img_fire_pixels

def plot_pixl_hist(distr, scale="linear"):
    """
    distr: list of number of fire pixels per image
    scale: "linear" or "log"
    """
    hist, bins, _ = plt.hist(distr, bins=len(distr))
    plt.title("Distribution of fire pixels")

    if scale == "log":
        plt.close()
        logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
        plt.hist(distr, bins=logbins)
        plt.xscale('log')
        plt.title("Distribution of fire pixels (log scale)")

    plt.xlabel("image count")
    plt.ylabel("fire pixel count")
    plt.show()

def split_in_patches(mask, N=16):
    """
    Splits the given image in N patches.
    :return: [list] N patches of equal size
    """
    mappings, img_size, patch_size = action_space_model('Landsat8')
    patches = []
    for i in range(N):
        buf = mappings[i]
        patch = mask[buf[0]:buf[0]+patch_size, buf[1]:buf[1]+patch_size]
        patches.append(patch)

    return patches

def get_label(num_fire_pixels, patch_size=64):
    """
    num_fire_pixels: [list] the number of fire pixels of each patch in an image
    :return: [ndarray] a vector of 1 where fire pixels are over threshold
             or 0 where they are under threshold
    """
    label = np.zeros(16)
    for i, num in enumerate(num_fire_pixels):
        fire_percentage = num/patch_size # * 100
        if fire_percentage > FIRE_THRES:
            label[i] = 1
        else:
            label[i] = 0
    return label
