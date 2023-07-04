import pandas as pd
import numpy as np
import warnings
import pickle
from scipy.io import loadmat

from torch.utils.data.dataset import Dataset
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore', Image.DecompressionBombWarning)

class CustomDatasetFromImages(Dataset):
    def __init__(self, csv_path, transform):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Transforms
        self.transforms = transform
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=None)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 1])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 0])
        # Calculate len
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Open image
        img_as_img = Image.open(single_image_name)
        # Transform the image
        img_as_tensor = self.transforms(img_as_img)
        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        return (img_as_tensor, single_image_label)

    def __len__(self):
        return self.data_len

class LandsatDataset(Dataset):
    def __init__(self, data_path):
        # data loading
        with open(data_path, 'rb') as f:
            dataset = pickle.load(f)

        self.data = dataset["data"]
        self.targets = dataset["targets"]
        self.num_examples = len(self.data)
    def __getitem__(self, index):
        return self.data[index], self.targets[index]
    def __len__(self):
        return self.num_examples


class LandsatSubset(Dataset):
    # this subset contains 16-4 selected images of the dataset that contain fire
    def __init__(self, data_path):

        if data_path == 'data/test15.pkl':
            with open('data/train85.pkl', 'rb') as f:
                dataset = pickle.load(f)

            data = np.empty((4, 256, 256, 3))
            targets = np.empty((4, 256, 256, 1))
            idx_fire_present = [69, 70, 78, 82]
            indexes = np.arange(0, len(idx_fire_present))
            for i, k in zip(indexes, idx_fire_present):
                data[i] = dataset["data"][k]
                targets[i] = dataset["targets"][k]
        else:
            # data loading
            with open(data_path, 'rb') as f:
                dataset = pickle.load(f)

            data = np.empty((16,256,256,3))
            targets = np.empty((16,256,256,1))
            idx_fire_present = [1, 4, 6, 9, 10, 13, 19, 21, 26, 33, 38, 42, 55, 58, 66, 67]
            indexes = np.arange(0,len(idx_fire_present))
            for i, k in zip(indexes, idx_fire_present):
                data[i] = dataset["data"][k]
                targets[i] = dataset["targets"][k]

        self.data = data
        self.targets = targets
        self.num_examples = len(self.data)
    def __getitem__(self, index):
        return self.data[index], self.targets[index]
    def __len__(self):
        return self.num_examples