"""
This function pretrains the high and low resolution classifiers.
How to run on different benchmarks:
    python classifer_training.py --model R32_C10, R32_C100, R34_fMoW, R50_ImgNet
       --lr 1e-1 (Different learning rates should be used for different benchmarks)
       --cv_dir checkpoint directory
       --batch_size 128
       --img_size 32, 224, 8, 56
"""
import os
from tensorboard_logger import configure, log_value
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.utils.data as torchdata
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
import torch.optim as optim
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
cudnn.benchmark = True

from utils import utils

TF_CPP_MIN_LOG_LEVEL="2"

import argparse
parser = argparse.ArgumentParser(description='Classifier Training')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--beta', type=float, default=1e-1, help='entropy multiplier')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
parser.add_argument('--model', default='R32_C10', help='R<depth>_<dataset> see utils.py for a list of configurations')
parser.add_argument('--data_dir', default='data/', help='data directory')
parser.add_argument('--load', help='checkpoint directory for trained model')
parser.add_argument('--cv_dir', default='cv/tmp/', help='checkpoint directory (models and logs are saved here)')
parser.add_argument('--batch_size', type=int, default=256, help='batch size') # LARGE BATCH SIZE CAUSES MEMORY ISSUE ON GPU
parser.add_argument('--epoch_step', type=int, default=10000, help='epochs after which lr is decayed')
parser.add_argument('--max_epochs', type=int, default=300, help='maximum number of epochs')
parser.add_argument('--parallel', action ='store_true', default=False, help='use multiple GPUs for training')
parser.add_argument('--test_interval', type=int, default=5, help='At what epoch to test the model')
parser.add_argument('--img_size', type=int, default=32, help='image size for the classification network')
parser.add_argument('--mode', default='hr', help='Type of the classifier - LR or HR')
args = parser.parse_args()

if not os.path.exists(args.cv_dir):
    os.system('mkdir ' + args.cv_dir)
utils.save_args(__file__, args)

def train(epoch):
    rnet.train()
    matches, losses = [], []
    for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(trainloader), total=len(trainloader)):

        inputs, targets = Variable(inputs), Variable(targets)#.cuda(non_blocking=True)
        # print("labels:", targets[:10])
        # print(inputs[0].shape) # 3x32x32 cifar
        # print(inputs[1].permute(1, 2, 0).shape) # 32x32x3
        if (args.model == 'R32_Landsat-8'):
            inputs = inputs.float().permute(0,3,1,2)
            targets = torch.Tensor([int(val) for val in targets])
            targets = targets.type(torch.LongTensor)  # casting to long
        if args.parallel:
            inputs = inputs.cuda()
            targets = targets.cuda()
            # inputs, targets = inputs.to(device), targets.to(device)

        inputs = torch.nn.functional.interpolate(inputs, (args.img_size, args.img_size))
        # for i in range(5):
        #     plt.imshow( inputs[i].cpu().permute(1, 2, 0) )
        #     plt.show()
        preds = rnet.forward(inputs, args.model.split('_')[1], args.mode)

        _, pred_idx = preds.max(1)
        match = (pred_idx==targets).data

        loss = F.cross_entropy(preds, targets)
 
        optimizer.zero_grad()
        loss.backward() # reduce image size sto ImageNet?? gia na min exw CUDA error: CUBLAS_STATUS_NOT_INITIALIZED
        optimizer.step()

        matches.append(match.cpu())
        losses.append(loss.cpu())

    # Compute training indicators
    accuracy = torch.cat(matches, 0).float().mean()
    loss = torch.stack(losses).mean()

    # Save the logs
    log_str = 'Epoch: %d | Acc: %.3f | Loss: %.3f'%(epoch, accuracy, loss)
    print(log_str)
    log_value('train_accuracy', accuracy, epoch)
    log_value('train_loss', loss, epoch)

def test(epoch):
    rnet.eval()
    matches = []
    for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(testloader), total=len(testloader)):

        inputs, targets = Variable(inputs, volatile=True), Variable(targets)#.cuda(non_blocking=True)
        if (args.model == 'R32_Landsat-8'):
            inputs = inputs.float().permute(0,3,1,2)
            targets = torch.Tensor([int(val) for val in targets])
            targets = targets.type(torch.LongTensor)  # casting to long
        if args.parallel:
            inputs = inputs.cuda()
            targets = targets.cuda()

        inputs = torch.nn.functional.interpolate(inputs, (args.img_size, args.img_size))

        preds = rnet.forward(inputs, args.model.split('_')[1], args.mode)

        _, pred_idx = preds.max(1)
        match = (pred_idx==targets).data

        matches.append(match.cpu())

    # Save the logs
    accuracy = torch.cat(matches, 0).float().mean()
    log_str = '[test] Epoch: %d | Acc: %.3f'%(epoch, accuracy)
    print(log_str)
    log_value('test_accuracy', accuracy, epoch)

    # Save the model parameters
    rnet_state_dict = rnet.state_dict()  # rnet.module.state_dict() if args.parallel else rnet.state_dict()
    state = {
      'state_dict': rnet_state_dict,
      'epoch': epoch,
      'acc': accuracy
    }
    torch.save(state, args.cv_dir+'Landsat-8/LR_ckpt_E_%d_A_%.3f'%(epoch, accuracy))

#--------------------------------------------------------------------------------------------------------#
import time
trainset, testset = utils.get_dataset(args.model, args.data_dir)
trainloader = torchdata.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0) # (was 8) set it to 0 to run debugger
testloader = torchdata.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0) # greater the num_workers, more efficiently the CPU load data and less the GPU has to wait

# Load the Model
rnet, _, _ = utils.get_model(args.model)
# if args.mode == "lr":
#     rnet = rnet_lr
# else:
#     rnet = rnet_hr
# doesn't matter cause they are the same

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.parallel:
    # rnet.to(device)
    rnet.cuda()

# Load the pre-trained classifier
if args.load:
    checkpoint = torch.load(args.load)
    rnet.load_state_dict(checkpoint['state_dict'])

# Save the configuration to the output directory
configure(args.cv_dir+'/log', flush_secs=5)

# Define the optimizer
if args.model.split('_')[1] == 'C10' or args.model.split('_')[1] == 'C100':
    optimizer = optim.SGD(rnet.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [150, 250])
else:
    optimizer = optim.Adam(rnet.parameters(), lr=args.lr)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [args.epoch_step])

print("Start training and testing...")
# Start training and testing
for epoch in range(args.max_epochs):
    train(epoch)
    if epoch % args.test_interval == 0:
        test(epoch)
    lr_scheduler.step()
