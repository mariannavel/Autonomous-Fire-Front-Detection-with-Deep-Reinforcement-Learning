import numpy as np
import warnings
import pickle
from torch.utils.data.dataset import Dataset
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore', Image.DecompressionBombWarning)


class LandsatDataset(Dataset):
    def __init__(self, data_path):
        # data loading
        with open(data_path, 'rb') as f:
            dataset = pickle.load(f)

        self.data = dataset["data"]
        self.targets = dataset["targets"]
        self.num_examples = len(self.data)
        # self.transforms = transform

    def __getitem__(self, index):
        return self.data[index], self.targets[index]
        # tensor_data = self.transforms(self.data[index])
        # return tensor_data, self.targets[index]

    def __len__(self):
        return self.num_examples


class LandsatSubset(Dataset):
    # this subset contains 16-4 selected images of the dataset that contain fires
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