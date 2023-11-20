import numpy as np
import rasterio
import torch
import torch.nn as nn
from models.unet_models_keras import get_model_keras
from models.unet_models_pytorch import get_model_pytorch
from utils.visualize import visualize_img3c_mask

MAX_PIXEL_VALUE = 65535 # used to normalize the image
IMAGE_SIZE = (256, 256)
N_CHANNELS = 3
N_FILTERS = 16
TH_FIRE = 0.25
WEIGHTS_FILE = 'unet_voting_final_weights.h5'
MASK_ALGORITHM = 'voting'
ARCHITECTURE = 'unet_{}f_2conv_{}'.format(N_FILTERS, '10c' if N_CHANNELS == 10 else '762' )

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

def test_unet(img_path, pytorch_unet):
    img3c = torch.from_numpy(get_img_762bands(img_path)) # in 3 channels
    # y_pred = keras_unet.predict(np.array([img3c]), batch_size=1)
    img3c = torch.unsqueeze(img3c, 0).permute(0, 3, 1, 2)
    y_pred = pytorch_unet.forward(img3c)
    visualize_img3c_mask(img3c[0], y_pred[0])

def dice_coefficient(predicted, target, smooth=1):
    """
    :param predicted: the predicted segmentation mask
    :param smooth: Smoothing factor (in [0, 1] to avoid division by zero
    :return: the dice coefficient between predicted and target
    """
    dice = torch.zeros(predicted.shape[0])
    # iterate a batch of segmentation masks
    for i, (p, t) in enumerate(zip(predicted, target)):
        intersection = torch.sum(p * t)
        union = torch.sum(p) + torch.sum(t)
        dice[i] = (2. * intersection + smooth) / (union + smooth)
    return dice

def IoU(predicted, target):
    """
    :return: Intersection over Union aka the Jaccard index
    """
    iou = torch.zeros(predicted.shape[0])
    # Calculate the intersection and union of the two binary masks
    for i, (p, t) in enumerate(zip(predicted, target)):
        intersection = torch.sum(p * t)
        union = torch.sum(p) + torch.sum(t)
        iou[i] = intersection / union

    # intersection = np.logical_and(predictions, targets)
    # union = np.logical_or(predictions, targets)

    # iou = np.sum(intersection) / np.sum(union)

    return iou

def compute_reward(preds, targets, policy, device):
    """
    :param preds: the predicted segmentation masks
    :param targets: the ground truth segmentation masks
    :param policy: binary vector indicating sampled or non-sampled patches
    """

    patch_use = policy.sum(1).float() / policy.size(1) # I will penalize the policy with the number of sampled patches
    sparse_reward = 1.0 - patch_use**2

    dice = dice_coefficient(preds, targets) # batch_size x 1
    for dc in dice: # assert dc >= 0 and dc <= 1
        torch._assert(dc >= 0 and dc <= 1, "Dice coefficient out of range [0,1] !")

    penalty = 1-dice # the segmentation error between 0 and 1

    reward = sparse_reward - penalty.to(device)

    return reward.unsqueeze(1), dice

def get_prediction(images, unet, device):
    TH_FIRE = 0.25
    # keras_unet = get_model_keras(model_name='unet',
    #                              input_height=IMAGE_SIZE[0], input_width=IMAGE_SIZE[1],
    #                              n_filters=N_FILTERS, n_channels=N_CHANNELS)
    # keras_unet.load_weights(WEIGHTS_FILE)
    y_pred = unet.forward(images) #, batch_size=args.batch_size)
    # y_pred = y_pred[:, :, :, 0] > TH_FIRE # edw ginontai binary oi times
    y_pred = y_pred > TH_FIRE
    # pred_masks = np.array(y_pred * 255, dtype=np.uint8)
    return (y_pred*1).to(device)


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