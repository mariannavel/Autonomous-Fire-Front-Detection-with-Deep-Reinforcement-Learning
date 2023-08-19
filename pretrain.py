"""
This file trains the Policy Network standalone with custom targets.
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm
import torch.optim as optim
from utils import utils
from utils.custom_dataloader import LandsatDataset
from tensorboard_logger import configure, log_value

import argparse
parser = argparse.ArgumentParser(description='Policy Network Pre-training')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--model', default='ResNet_Landsat8')
parser.add_argument('--data_dir', default='data/', help='data directory')
parser.add_argument('--ckpt_dir', default='cv/tmp/pretrain', help='checkpoint directory (models and logs are saved here)')
parser.add_argument('--batch_size', type=int, default=32, help='batch size') # INCREASED batch size 8 --> 16 --> 32 --> 64 --(2K dset)--> SIGKILL
parser.add_argument('--max_epochs', type=int, default=500, help='total epochs to run')
parser.add_argument('--LR_size', type=int, default=32, help='Policy Network Image Size')
parser.add_argument('--test_interval', type=int, default=10, help='Every how many epoch to test the model')
parser.add_argument('--ckpt_interval', type=int, default=50, help='Every how many epoch to save the model')
args = parser.parse_args()

def train(model, epoch):

    losses, accuracies = [], []

    for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(trainloader), total=len(trainloader), disable=True):

        inputs = inputs.to(device)
        targets = targets.to(device)

        inputs = inputs.float().permute(0, 3, 1, 2)
        inputs_agent = torch.nn.functional.interpolate(inputs.clone(), (args.LR_size, args.LR_size))

        # Run LR image through Policy Network
        probs = model.forward(inputs_agent)

        preds = probs.type(torch.DoubleTensor)
        preds[preds < 0.5] = 0.0
        preds[preds >= 0.5] = 1.0

        loss = BCE(preds, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # accuracy = sum(preds == targets)/len(preds)
        matches = (preds == targets).int()
        accuracy = matches.sum(dim=1).sum() / (matches.shape[0] * matches.shape[1]) # ??

        losses.append(loss)
        accuracies.append(accuracy)

    avg_loss = sum(losses)/len(losses)
    avg_accuracy = sum(accuracies)/len(accuracies)
    log_value(f'train_loss', avg_loss, epoch)
    log_value(f'train_accuracy', avg_accuracy, epoch)
    print(f"Epoch {epoch} | loss: {avg_loss} | accuracy: {avg_accuracy}")


def test(model, epoch):

    for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(trainloader), total=len(trainloader), disable=True):

        inputs = inputs.to(device)
        targets = targets.to(device)

        inputs = inputs.float().permute(0, 3, 1, 2)
        inputs_agent = torch.nn.functional.interpolate(inputs.clone(), (args.LR_size, args.LR_size))

        # Run LR image through Policy Network
        probs = model.forward(inputs_agent)

        # Save checkpoint!!!

if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ckpt_dir = 'checkpoints/pretrain/logs'
    if not os.path.exists(ckpt_dir): # change to --ckpt_dir
        os.system('mkdir ' + ckpt_dir)
    # Save the log values to the checkpoint directory
    configure(ckpt_dir, flush_secs=5)

    # Load the images and targets
    trainset = LandsatDataset('data/train_agent85.pkl')
    testset = LandsatDataset('data/test_agent15.pkl')

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Load the policy network model
    model = utils.get_model(args.model)
    model.to(device)

    BCE = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 1

    print('--- Policy Network Pre-training ---')

    for epoch in range(start_epoch, start_epoch+args.max_epochs):
        train(model, epoch)
        if epoch % args.test_interval == 0:
            test(model, epoch)