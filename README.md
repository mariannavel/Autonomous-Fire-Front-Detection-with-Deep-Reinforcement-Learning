# Autonomous Fire Front Detection in Satellite Images with Deep Reinforcement Learning

Files and directories:

- /dataset: explore and prepare the data for training, perform custom labeling
- /models: Policy Network and U-net models in PyTorch and Keras
- /utils: functions to get and save data of the agent, utilities for segmentation, visualization-plots, dataloader for Landsat-8
- train_agent_single.py: train the RL agent with REINFORCE, modeling trajectories with single-step MDPs
- train_agent_multi.py: train the RL agent with REINFORCE, modeling trajectories with multi-step MDPs (ongoing)
- train_multilabel.py: train PN with custom (binary vector) labels - can be perceived as a pretraining step for the agent
- inference.py: perform agent/U-net inference, forward data to the deep RL fire detection system, up-sampling & stochastic sampling experiments
- load_stats.py: load train/test stats to produce figures

To train the agent and form policy from scratch, you must run the following:

python train_agent_single.py --model ResNet
       --lr 1e-3
       --batch_size 64
       --LR_size 8, 16, 32
       --max_epochs 1000
       --data_dir 'dataset/data/'
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
