import torch
from utils import utils
from utils.custom_dataloader import LandsatDataset
import numpy as np
from unet.unet_models_pytorch import get_model_pytorch
from segnet import *
from utils.visualize import visualize_image, visualize_images

LR_size = 32 # PN input dims
NUM_SAMPLES = 1000

def inference(model, images, mappings, patch_size):
    """ Get the agent masked images from input,
    based on the policy distribution output """

    images = images.permute(0, 3, 1, 2)

    # Get the low resolution agent images
    input_agent = images.clone()
    input_agent = torch.nn.functional.interpolate(input_agent, (LR_size, LR_size))

    # get the probability distribution output of PN
    # probs = torch.sigmoid(PolicyNetwork(input_agent))
    probs = model(input_agent)

    # Sample the test-time policy
    policy = probs.data.clone()
    policy[policy < 0.5] = 0.0
    policy[policy >= 0.5] = 1.0

    # Get the masked high-res image and perform inference
    masked_image = utils.get_agent_masked_image(images, policy, mappings, patch_size)

    return masked_image

def load_images_PD_SIS(datapath, num=0):
    """ num: in [1, 21] number of images from the selected Landsat-8 dataset
        return: num tensor images """

    # fire_idx = [1, 4, 6, 9, 10, 13, 19, 21, 26, 33, 38, 42, 55, 58, 66, 67, 69, 70, 78, 80, 82]

    # ndarray: n x 256 x 256 x 3
    dataset = LandsatDataset(datapath)

    # get num random images from "fired images" set
    # indexes = np.random.randint(0, len(fire_idx), num)
    indexes = np.arange(0, len(dataset.data)) # len(fire_idx)

    ret_data = torch.empty((len(indexes), 256, 256, 3))
    ret_masks = torch.empty((len(indexes), 256, 256, 1))
    for i in indexes:
        ret_data[i] = torch.from_numpy(dataset.data[i])
        ret_masks[i] = torch.from_numpy(dataset.targets[i])

    return ret_data, ret_masks

def load_images_PN(datapath):
    # ndarray: n x 256 x 256 x 3
    dataset = LandsatDataset(datapath)

    indexes = np.arange(0, len(dataset.data))  # len(dataset.data)

    ret_data = torch.empty((len(indexes), 256, 256, 3))
    ret_labels = torch.empty((len(indexes), 16))
    for i in indexes:
        ret_data[i] = torch.from_numpy(dataset.data[i])
        ret_labels[i] = torch.from_numpy(dataset.targets[i])

    return ret_data, ret_labels

def get_model_results(images, model, ckpt_path, device, mode="Test"):

    # Load model and weights
    PolicyNetwork = utils.get_model(model)
    state_dict = torch.load(ckpt_path, map_location=torch.device('cpu'))
    PolicyNetwork.load_state_dict(state_dict["agent"])

    # Get agent action space
    mappings, _, patch_size = utils.action_space_model('Landsat8')

    masked_images = inference(PolicyNetwork, images, mappings, patch_size).to(device)

    # Transform before visualization
    masked_images = masked_images.permute(0,2,3,1).cpu().numpy()
    # save the images to feed them to a U-Net later...
    # np.save("pretrainPN/masked_images", masked_images)

    # visualize_images(images,
    #              masked_images,
    #              title=mode+" image: original vs masked",
    #              savepath="pretrainPN/inference/")

    return masked_images

def compute_metrics(preds, targets):
    dc = dice_coefficient(preds, targets)
    iou = IoU(preds, targets)
    return dc, iou

def thres_exp_main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    thresval = 0.04

    # Load images on which the policy network was NOT trained
    images, _ = load_images_PN(
        datapath=f'pretrainPN/threshold_experiment/{NUM_SAMPLES}/thres{thresval}/data/test.pkl')

    # Feed them to PN and save the produced masks
    masked_images = get_model_results(images,
                                      model='ResNet_Landsat8',
                                      ckpt_path=f"pretrainPN/threshold_experiment/{NUM_SAMPLES}/thres{thresval}/checkpoints/PN_pretrain_E_20_F1_0.000",
                                      device=device,
                                      mode="Test")

    # load masked images as tensors
    # masked_images = torch.from_numpy(np.load("pretrainPN/masked_images.npy")).permute(0,3,1,2)
    masked_images = torch.from_numpy(masked_images).permute(0, 3, 1, 2)

    # load U-Net
    unet = get_model_pytorch(model_name='unet', n_filters=16, n_channels=3)
    # load weights
    unet.load_state_dict(torch.load('train_agent/Landsat-8/unet/pytorch_unet.pt'))
    unet.eval()

    # Give masked images to U-Net and get their segment. mask
    batch_size = 32
    masked_batches = [masked_images[i:i + batch_size] for i in range(0, len(masked_images), batch_size)]
    preds = []  # compute in batches because of memory limitations
    for batch in masked_batches:
        with torch.no_grad():
            batch_preds = get_SegNet_prediction(batch, unet, device)
        preds.append(batch_preds)
    final_preds = torch.cat(preds)
    # preds = get_SegNet_prediction(masked_images, unet, device)
    # visualize_image(preds.cpu().permute(0,2,3,1))

    # Load the target segment. masks of the original images
    _, targets = load_images_PD_SIS(datapath=f'data/{NUM_SAMPLES}/test.pkl')
    targets = targets.permute(0, 3, 1, 2).to(device)

    # Evaluate test performance
    dc, iou = compute_metrics(final_preds, targets)

    print("--- Evaluation of the custom-label ResNet on 15 samples ---")
    print(f"Dice: {torch.mean(dc)} | IoU: {torch.mean(iou)}")


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    images, targets = load_images_PD_SIS(datapath=f'data/{NUM_SAMPLES}/test.pkl')
    targets = targets.permute(0, 3, 1, 2).to(device)

    # Feed them to PN and save the produced masks
    masked_images = get_model_results(images,
                    model='ResNet18_Landsat8',
                    ckpt_path=f"train_agent/{NUM_SAMPLES}/batch_sz_256_LR_32/checkpoints/Policy_ckpt_E_1000_R_0.402_Res",
                    device=device,
                    mode="Test")

    # load masked images as tensors
    # masked_images = torch.from_numpy(np.load("pretrainPN/masked_images.npy")).permute(0,3,1,2)
    masked_images = torch.from_numpy(masked_images).permute(0,3,1,2)

    # load U-Net
    unet = get_model_pytorch(model_name='unet', n_filters=16, n_channels=3)
    # load weights
    unet.load_state_dict(torch.load('train_agent/Landsat-8/unet/pytorch_unet.pt'))
    unet.eval()

    # Give masked images to U-Net and get their segment. mask
    # batch_size = 32
    # masked_batches = [masked_images[i:i + batch_size] for i in range(0, len(masked_images), batch_size)]
    # preds = [] # compute in batches because of memory limitations
    # for batch in masked_batches:
    #     with torch.no_grad():
    #         batch_preds = get_SegNet_prediction(batch, unet, device)
    #     preds.append(batch_preds)
    # final_preds = torch.cat(preds)
    preds = get_SegNet_prediction(masked_images, unet, device)
    # visualize_image(preds.cpu().permute(0,2,3,1))

    # Evaluate test performance
    dc, iou = compute_metrics(preds, targets)

    print("--- Evaluation of the custom-label ResNet on 15 samples ---")
    print(f"Dice: {torch.mean(dc)} | IoU: {torch.mean(iou)}")
