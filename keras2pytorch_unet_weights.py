import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Seg_models import get_model_keras
from Seg_models_pytorch import get_model_pytorch

IMAGE_SIZE = (256, 256)
N_CHANNELS = 3
N_FILTERS = 16
WEIGHTS_FILE ='cv/tmp/Landsat-8/unet/model_unet_voting_final_weights.h5'

keras_unet = get_model_keras(model_name='unet',
                 input_height=IMAGE_SIZE[0], input_width=IMAGE_SIZE[1],
                 n_filters=N_FILTERS, n_channels=N_CHANNELS)
# keras_unet.summary()
keras_unet.load_weights(WEIGHTS_FILE)

pytorch_unet = get_model_pytorch(model_name='unet', n_filters=N_FILTERS, n_channels=N_CHANNELS)

keras_layers = [] # list of learnable keras layers
for layer in keras_unet.layers:
    if len(layer.get_weights()) > 0:
        # print(f'keras weight {layer.get_weights()[0].shape}')
        keras_layers.append(layer)

print("Keras layers:", len(keras_layers)) # 41
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
        count_conv2d += 1 # 23

    elif isinstance(layer, nn.BatchNorm2d):
        # Copy the weights from Keras to PyTorch
        layer.weight.data = torch.from_numpy(keras_layers[k].get_weights()[0])
        layer.bias.data = torch.from_numpy(keras_layers[k].get_weights()[1])
        layer.running_mean.data = torch.from_numpy(keras_layers[k].get_weights()[2])
        layer.running_var.data = torch.from_numpy(keras_layers[k].get_weights()[3])
        k += 1
        count_bn += 1 # 18

# Save the PyTorch model's state dictionary
torch.save(pytorch_unet.state_dict(), 'pytorch_unet.pt')