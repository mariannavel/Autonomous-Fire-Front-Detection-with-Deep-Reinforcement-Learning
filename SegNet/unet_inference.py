from SegNet.unet_models_keras import get_model_keras
from utils import *
import torch

MASK_ALGORITHM = 'voting'
ARCHITECTURE = 'unet_{}f_2conv_{}'.format(N_FILTERS, '10c' if N_CHANNELS == 10 else '762' )

WEIGHTS_FILE = 'cv/tmp/Landsat-8/unet/model_unet_{}_final_weights.h5'.format(MASK_ALGORITHM)

def test_unet(img_path, pytorch_unet):
    img3c = torch.from_numpy(get_img_762bands(img_path)) # in 3 channels
    # y_pred = keras_unet.predict(np.array([img3c]), batch_size=1)
    img3c = torch.unsqueeze(img3c, 0).permute(0, 3, 1, 2)
    y_pred = pytorch_unet.forward(img3c)
    visualize_with_seg_mask(img3c[0], y_pred[0])

def generate_inference(img_path='data/images', msk_path='data/voting_masks'):
    """ Warning: This will generate and plot masks of the WHOLE dataset"""

    # 1. Load the model architecture (Unet-Light-3c)
    model = get_model_keras(model_name='unet', input_height=IMAGE_SIZE[0], input_width=IMAGE_SIZE[1], n_filters=N_FILTERS,
                      n_channels=N_CHANNELS)
    # model.summary()

    # 2. Load the weights (trained on the voting scheme)
    model.load_weights(WEIGHTS_FILE)
    print('Weights Loaded!')

    img_filelist = sorted(os.listdir(img_path))
    msk_filelist = sorted(os.listdir(msk_path))
    for i, (fn_img, fn_mask) in enumerate( zip(img_filelist, msk_filelist )):
        # 3. Load the image to be segmented
        img_path = os.path.join(img_path, fn_img)
        img3c = get_img_762bands(img_path) # in 3 channels

        mask = get_mask_arr(os.path.join(msk_path, fn_mask))

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
