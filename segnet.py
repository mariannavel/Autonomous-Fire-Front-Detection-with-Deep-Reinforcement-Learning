import torch
import numpy as np

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

def compute_SegNet_reward(preds, targets, policy, device):
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

def get_SegNet_prediction(images, unet, device):
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