import numpy as np
import rasterio
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from SegNet.unet_models_keras import get_model_keras
from SegNet.unet_models_pytorch import get_model_pytorch

MAX_PIXEL_VALUE = 65535 # Max. pixel value, used to normalize the image
IMAGE_SIZE = (256, 256)
N_CHANNELS = 3
N_FILTERS = 16
TH_FIRE = 0.25

WEIGHTS_FILE = '../cv/tmp/Landsat-8/unet/unet_voting_final_weights.h5'

def get_img_arr(path):
    img = rasterio.open(path).read().transpose((1, 2, 0))
    img = np.float32(img)/MAX_PIXEL_VALUE
    return img

def get_mask_arr(path):
    img = rasterio.open(path).read().transpose((1, 2, 0))
    seg = np.float32(img)
    return seg

def get_img_762bands(path):
    img = rasterio.open(path).read((7, 6, 2)).transpose((1, 2, 0))
    img = np.float32(img) / MAX_PIXEL_VALUE
    return img

def visualize_dataset3c(path='data/images'):
    # iterate over files in that path
    for i, filename in enumerate(os.listdir(path)):
        img = os.path.join(path, filename)
        img3c = get_img_762bands(img) # 3 channels
        plt.imshow(img3c)
        plt.title("Image"+str(i))
        plt.show()

def visualize_with_seg_mask(img3c, mask):
    # permute for visualization purposes
    img3c = img3c.float().permute(2, 1, 0)
    mask = mask.float().permute(2, 1, 0)
    # mask = torch.unsqueeze(mask[0] > 0, dim=0) # make it binary
    mask = mask > TH_FIRE
    plt.subplot(1, 2, 1)
    plt.imshow(img3c.detach().numpy())
    plt.title('Original image 3c')

    plt.subplot(1, 2, 2)
    plt.imshow(mask.detach().numpy())
    plt.title('Voting mask (target)')

    plt.show()

def keras2pytorch_model(keras_model, pytorch_model):
    keras_unet = get_model_keras(model_name='unet',
                                 input_height=IMAGE_SIZE[0], input_width=IMAGE_SIZE[1],
                                 n_filters=N_FILTERS, n_channels=N_CHANNELS)
    # keras_unet.summary()
    keras_unet.load_weights(WEIGHTS_FILE)

    pytorch_unet = get_model_pytorch(model_name='unet', n_filters=N_FILTERS, n_channels=N_CHANNELS)

    keras_layers = []  # list of learnable keras layers
    for layer in keras_unet.layers:
        if len(layer.get_weights()) > 0:
            # print(f'keras weight {layer.get_weights()[0].shape}')
            keras_layers.append(layer)

    print("Keras layers:", len(keras_layers))  # 41
    # PyTorch layers: 83
    k = 0
    count_conv2d = 0
    count_bn = 0

    for layer in pytorch_unet.modules():

        # Copy the weights from Keras to PyTorch - only for learnable layers

        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
            weights = keras_layers[k].get_weights()
            layer.weight.data = torch.from_numpy(np.transpose(weights[0]))
            layer.bias.data = torch.from_numpy(np.transpose(weights[1]))
            k += 1
            count_conv2d += 1  # 23

        elif isinstance(layer, nn.BatchNorm2d):
            # Copy the weights from Keras to PyTorch
            layer.weight.data = torch.from_numpy(keras_layers[k].get_weights()[0])
            layer.bias.data = torch.from_numpy(keras_layers[k].get_weights()[1])
            layer.running_mean.data = torch.from_numpy(keras_layers[k].get_weights()[2])
            layer.running_var.data = torch.from_numpy(keras_layers[k].get_weights()[3])
            k += 1
            count_bn += 1  # 18

    # Save the PyTorch model's state dictionary
    torch.save(pytorch_unet.state_dict(), 'pytorch_unet.pt')