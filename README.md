# PatchDrop with Semantic Image Segmentation (PD-SIG)

To train the agent in order to form policy from scratch, you must run the following:

python train_agent.py --model ResNet_Landsat8
       --lr 1e-3
       --batch_size 64
       --LR_size 8, 16, 32
       --max_epochs 1000
       --data_dir 'data/'
       --cv_dir 'checkpoints/'
       --test_interval 10
       --ckpt_interval 100

Adjust the parameters as you wish depending on the benchmark. Remember to set the variables CKPT_UNET and NUM_SAMPLES before each run.

Parameters:
-----------
**model**: 
- ResNet_Landsat8 is a simple ResNet of 4 layers, each having 1 convolutional block. This is used when we train with few data (<1000).
- ResNet18_Landsat8 is a ResNet18 (4 layers of 2 blocks each). Used when training with more data.
- CNN_Landsat8 to run pipeline with a simple CNN of 3 layers as policy network.

**data_dir**: the data directory

**cv_dir**: the checkpoint directory (models and logs are saved here)

**load**: checkpoint to load pretrained agent

**lr**: the learning rate

**batch_size**: the higher is better when you have a lot of data

**max_epochs**: num of epochs to run

**alpha**: the probability bounding factor

**LR_size**: the low-res input size of policy network

**test_interval**: every how many epochs to test the model

**ckpt_interval**: every how many epochs to save the model
