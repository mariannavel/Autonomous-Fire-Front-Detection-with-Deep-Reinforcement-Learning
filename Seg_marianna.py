from Seg_generator import *
import matplotlib.pyplot as plt
import os
from Seg_models import get_model_keras

DATA_PATH = 'data/images'
MSK_PATH = 'data/voting_masks'

IMAGE_SIZE = (256, 256)
N_CHANNELS = 3
N_FILTERS = 16
TH_FIRE = 0.25

MASK_ALGORITHM = 'voting'
ARCHITECTURE = 'unet_{}f_2conv_{}'.format(N_FILTERS, '10c' if N_CHANNELS == 10 else '762' )

WEIGHTS_FILE = 'cv/tmp/Landsat-8/unet/model_unet_{}_final_weights.h5'.format(MASK_ALGORITHM)

print('Images at: {}'.format(DATA_PATH))

def visualize_dataset3c():
    # iterate over files in that path
    for i, filename in enumerate(os.listdir(DATA_PATH)):
        img = os.path.join(DATA_PATH, filename)
        # print(i, img)
        img3c = get_img_762bands(img) # 3 channels
        plt.imshow(img3c)
        plt.title("Image"+str(i))
        plt.show()

def generate_inference():
    """ Warning: This will generate and plot masks of the WHOLE dataset"""

    # 1. Load the model architecture (Unet-Light-3c)
    model = get_model_keras(model_name='unet', input_height=IMAGE_SIZE[0], input_width=IMAGE_SIZE[1], n_filters=N_FILTERS,
                      n_channels=N_CHANNELS)
    # model.summary()

    # 2. Load the weights (trained on the voting scheme)
    model.load_weights(WEIGHTS_FILE)
    print('Weights Loaded!')

    img_filelist = sorted(os.listdir(DATA_PATH))
    msk_filelist = sorted(os.listdir(MSK_PATH))
    for i, (fn_img, fn_mask) in enumerate( zip(img_filelist, msk_filelist )):
        # 3. Load the image to be segmented
        img_path = os.path.join(DATA_PATH, fn_img)
        img3c = get_img_762bands(img_path) # in 3 channels

        mask = get_mask_arr(os.path.join(MSK_PATH, fn_mask))

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

        # 4. Repeat for all the images of the dataset

generate_inference()