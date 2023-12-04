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

Adjust the parameters depending on the benchmark. Remember to set the variables CKPT_UNET and NUM_SAMPLES before each run.

Parameters:

**model**: CNN, ResNet (a simple ResNet of 12 layers total), ResNet18

**data_dir**: the data directory

**cv_dir**: the checkpoint directory (models and logs are saved here)

**load**: checkpoint to load pretrained agent

**lr**: the learning rate

**batch_size**: higher batch size is better when you have a lot of data

**max_epochs**: number of epochs to run

**alpha**: the probability bounding factor

**LR_size**: the low-res input size of policy network

**test_interval**: every how many epochs to test the model

**ckpt_interval**: every how many epochs to save the model
