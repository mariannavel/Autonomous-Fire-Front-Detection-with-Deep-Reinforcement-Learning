from utils import agent_utils
from utils.unet_utils import *
import matplotlib.pyplot as plt
from utils.visualize import plot_inference_grid
from utils.custom_dataloader import LandsatDataset
from models.unet_models_pytorch import get_model_pytorch
from models.unet_models_keras import get_model_keras
import pickle
import torch

LR_size = 32
HR_size = 256
NUM_SAMPLES = 100

device = 'cuda' if torch.cuda.is_available() else 'cpu'

mappings, _, patch_size = agent_utils.get_action_space()

def unet_inference(img_path, msk_path):

    # 1. Load the model architecture (Unet-Light-3c)
    model = get_model_keras(model_name='unet', input_height=IMAGE_SIZE[0], input_width=IMAGE_SIZE[1], n_filters=N_FILTERS,
                      n_channels=N_CHANNELS)
    # model.summary()

    # 2. Load the weights (trained on the voting scheme)
    model.load_weights(WEIGHTS_FILE)
    # print("Keras:", model.weights[0].shape)

    # 3. Load the image to be segmented
    img3c = get_img_762bands(img_path)  # in 3 channels
    mask = get_mask_arr(msk_path)

    y_pred = model.predict(np.array([img3c]), batch_size=1)
    y_pred = y_pred[0, :, :, 0] > TH_FIRE
    pred_mask = np.array(y_pred * 255, dtype=np.uint8)

    plt.subplot(1, 3, 1)
    plt.imshow(img3c)
    plt.title('Original image (3c)')

    plt.subplot(1, 3, 2)
    plt.imshow(pred_mask)
    plt.title('Predicted mask')

    plt.subplot(1, 3, 3)
    plt.imshow(mask)
    plt.title('Target')

    plt.show()

def agent_inference(model, images, mappings, patch_size):
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
    masked_image = agent_utils.get_masked_image(images, policy, mappings, patch_size)

    return masked_image


def load_images_labels(datapath, label_type, num=0):
    """
    :param datapath: path to load data from
    :param label_type: "masks" or "custom"
    :param num: [optional] number of images to sample from fire-present images set
    :return: images and labels as tensors
    """
    # ndarray: n x 256 x 256 x 3
    dataset = LandsatDataset(datapath)

    # indexes = np.random.randint(0, len(fire_idx), num)
    indexes = np.arange(0, len(dataset.data))  # len(dataset.data)

    ret_data = torch.empty((len(indexes), 256, 256, 3))
    if label_type == "custom":
        ret_labels = torch.empty((len(indexes), 16))
    elif label_type == "masks":
        ret_labels = torch.empty((len(indexes), 256, 256, 1))
    else:
        print("wrong label type")
        exit(1)

    for i in indexes:
        ret_data[i] = torch.from_numpy(dataset.data[i])
        ret_labels[i] = torch.from_numpy(dataset.targets[i])

    return ret_data, ret_labels

def get_policy_prediction(images, model, ckpt_path, device):

    # Load model and weights
    policy_net = agent_utils.get_model(model)
    state_dict = torch.load(ckpt_path, map_location=torch.device('cpu'))
    policy_net.load_state_dict(state_dict["agent"])

    masked_images = agent_inference(policy_net, images, mappings, patch_size).to(device)

    # Transform before visualization
    masked_images = masked_images.permute(0,2,3,1).cpu().numpy()
    # save the images to feed them to a U-Net later...
    # np.save("pretrainPN/masked_images", masked_images)

    return masked_images

def compute_metrics(preds, targets):
    dc = dice_coefficient(preds, targets)
    iou = IoU(preds, targets)
    return dc, iou

def pretrained_multi_label_forward():

    thresval = 0.05

    # Load images on which the policy network has NOT been trained
    images, _ = load_images_labels(
        datapath=f'pretrainResNet/{NUM_SAMPLES}/thres{thresval}/data/test.pkl', label_type='masks')

    masked_images = get_policy_prediction(images,
                                          model='ResNet',
                                          ckpt_path=f"pretrainResNet/{NUM_SAMPLES}/thres{thresval}/checkpoints/PN_pretrain_E_20_F1_0.000",
                                          device=device)

    # load masked images as tensors
    # masked_images = torch.from_numpy(np.load("pretrainPN/masked_images.npy")).permute(0,3,1,2)
    masked_images = torch.from_numpy(masked_images).permute(0, 3, 1, 2)

    # load U-Net
    unet = get_model_pytorch(model_name='unet', n_filters=16, n_channels=3)
    # load weights
    unet.load_state_dict(torch.load('train_agent/Landsat-8/unet/pytorch_unet.pt'))
    unet.eval()

    # Give masked images to U-Net and get their segmented mask
    batch_size = 32
    masked_batches = [masked_images[i:i + batch_size] for i in range(0, len(masked_images), batch_size)]
    preds = []  # compute in batches because of memory limitations
    for batch in masked_batches:
        with torch.no_grad():
            batch_preds = get_prediction(batch, unet, device)
        preds.append(batch_preds)
    final_preds = torch.cat(preds)
    # preds = get_prediction(masked_images, unet, device)

    # Load the target seg. masks of the original images
    _, targets = load_images_labels(datapath=f'data/{NUM_SAMPLES}/rand_sampled/test.pkl')
    targets = targets.permute(0, 3, 1, 2).to(device)

    # Evaluate test performance
    dc, iou = compute_metrics(final_preds, targets)

    print("--- Evaluation of the custom-label network on test samples ---")
    print(f"Dice: {torch.mean(dc)} | IoU: {torch.mean(iou)}")

def deepRL_forward():

    images, targets = load_images_labels(datapath=f'data/{NUM_SAMPLES}/stratified/test.pkl',
                                         label_type="masks")
    targets = targets.permute(0, 3, 1, 2).to(device)

    masked_images = get_policy_prediction(images,
                                          model='ResNet',
                                          ckpt_path=f"train_agent/{NUM_SAMPLES}/",
                                          device=device)

    masked_images = torch.from_numpy(masked_images).permute(0,3,1,2)

    # load U-Net
    unet = get_model_pytorch(model_name='unet', n_filters=16, n_channels=3)
    # load weights
    unet.load_state_dict(torch.load('train_agent/Landsat-8/unet/pytorch_unet.pt'))
    unet.eval()

    preds = get_prediction(masked_images, unet, device)

    # Evaluate test performance
    dc, iou = compute_metrics(preds, targets)

    print("--- Evaluation of the deep RL experiment on test samples ---")
    print(f"Dice: {torch.mean(dc)} | IoU: {torch.mean(iou)}")


def stochastic_sample_experiment(x_hr):

    with open(f"data/EDA/fp_dict{NUM_SAMPLES}.pkl", 'rb') as f:
        fire_distr = pickle.load(f)

    # Model the patch distribution
    probs = [value/NUM_SAMPLES for value in fire_distr.values()]

    choices = []
    # randomly sample as many times as there are images
    for _ in range(len(x_hr)):
        how_many = np.random.choice(a=list(fire_distr.keys()), size=1, p=probs)
        which = np.random.choice(a=np.arange(16), size=int(how_many[0]), p=None, replace=False) # sample uniformly
        # np.array of 16 selections --> to tensor
        one_hot_which = []
        for index in which:
            one_hot_which.append(torch.nn.functional.one_hot(torch.tensor(index), num_classes=16))

        one_hot_choices = one_hot_which[0].clone()
        for i in range(1, len(one_hot_which)):
            one_hot_choices += one_hot_which[i]

        choices.append(one_hot_choices.tolist())

    masked_image = agent_utils.get_masked_image(x_hr, torch.tensor(choices), mappings, patch_size)

    return masked_image

def up_sample_experiment(x_hr):
    # sub-sample
    x_lr = torch.nn.functional.interpolate(x_hr.clone(), (LR_size, LR_size))

    # up-sample
    x_hr_up = torch.nn.functional.interpolate(x_lr.clone(), (HR_size, HR_size))

    return x_hr_up

def run_experiments():

    x_hr, targets = load_images_labels(datapath=f'data/{NUM_SAMPLES}/stratified/test.pkl', label_type="masks")
    targets = targets.permute(0, 3, 1, 2)
    x_hr = x_hr.permute(0, 3, 1, 2)

    x_hr_up = up_sample_experiment(x_hr)
    x_hr_masked = stochastic_sample_experiment(x_hr)

    # load unet and weights
    unet = get_model_pytorch(model_name='unet', n_filters=16, n_channels=3)
    unet.load_state_dict(torch.load('models/weights/pytorch_unet.pt'))
    unet.eval()

    preds_up = get_prediction(x_hr_up, unet, device='cpu')
    preds_stochastic = get_prediction(x_hr_masked, unet, device='cpu')

    dc_up, iou_up = compute_metrics(preds_up, targets)
    dc_stochastic, iou_stochastic = compute_metrics(preds_stochastic, targets)

    print("-- Up-sampling Experiment --")
    print(f"Dice: {torch.mean(dc_up)} | IoU: {torch.mean(iou_up)}\n")
    print("-- Stochastic Sampling Experiment --")
    print(f"Dice: {torch.mean(dc_stochastic)} | IoU: {torch.mean(iou_stochastic)}")


if __name__ == "__main__":

    data_path = f'data/{NUM_SAMPLES}/stratified/test.pkl'
    ckpt_path = f"train_agent/{NUM_SAMPLES}/stratified/checkpoints/Policy_ckpt_E_4300_R_0.450_Res"


    images, targets = load_images_labels(datapath=data_path, label_type="masks")
    targets = targets.permute(0, 3, 1, 2)

    masked_images = get_policy_prediction(images, model='ResNet', ckpt_path=ckpt_path, device=device)

    # load masked images as tensors
    # masked_images = torch.from_numpy(np.load("pretrainPN/masked_images.npy")).permute(0,3,1,2)
    masked_images = torch.from_numpy(masked_images).permute(0,3,1,2)

    # indexes = [9, 12 , 73, 15, 19, 25, 28, 39, 47, 79, 89, 114, 130, 136, 160, 172, 197, 152, 74, 80]
    for i in range(len(images)):
        plot_inference_grid(images[i-1], masked_images[i-1].permute(1,2,0), targets[i-1].permute(1,2,0))
