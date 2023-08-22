"""
This file trains the Policy Network standalone with custom targets (saved as binary vectors .
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
from visualize import visualize_image
from sklearn.metrics import f1_score
import numpy as np

import argparse
parser = argparse.ArgumentParser(description='Policy Network Pre-training')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--model', default='ResNet_Landsat8')
parser.add_argument('--data_dir', default='data/', help='data directory')
parser.add_argument('--cv_dir', default='checkpoints', help='checkpoint directory (models and logs are saved here)')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--max_epochs', type=int, default=20, help='total epochs to run')
parser.add_argument('--LR_size', type=int, default=32, help='Policy Network Image Size')
parser.add_argument('--test_interval', type=int, default=10, help='Every how many epoch to test the model')
parser.add_argument('--ckpt_interval', type=int, default=10, help='Every how many epoch to save the model')
args = parser.parse_args()

utils.save_args(__file__, args)

def train(model, epoch):

    losses, accuracies, f1_scores = [], [], []

    for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(trainloader), total=len(trainloader), disable=True):

        inputs = inputs.to(device)
        targets = targets.to(device)

        inputs = inputs.float().permute(0, 3, 1, 2)
        inputs_agent = torch.nn.functional.interpolate(inputs.clone(), (args.LR_size, args.LR_size))

        # Run LR image through Policy Network
        logits = model.forward(inputs_agent, activation=False)

        preds = logits.type(torch.DoubleTensor).to(device)

        loss = criterion(preds, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = torch.sigmoid(preds)
            preds[preds < 0.5] = 0.0
            preds[preds >= 0.5] = 1.0
            matches = (preds == targets).int()
            accuracy = matches.sum(dim=1).sum() / (matches.shape[0] * matches.shape[1])
            micro_f1 = f1_score(targets.cpu().numpy(), preds.cpu().numpy(), average='micro',
                                labels=np.unique(preds.cpu().numpy()), zero_division=1)

        # for input in inputs:
        #     visualize_image(input.permute(1,2,0).cpu())
        losses.append(loss.item())
        accuracies.append(accuracy.item())
        f1_scores.append(micro_f1)

    avg_loss = sum(losses)/len(losses)
    avg_accuracy = sum(accuracies)/len(accuracies)
    avg_f1 = sum(f1_scores)/len(f1_scores)
    log_value(f'train_loss', avg_loss, epoch)
    log_value(f'train_accuracy', avg_accuracy, epoch)
    log_value(f'train_F1_score', avg_f1, epoch)
    print(f"Epoch %d | loss: %.4f | accuracy: %.4f | F1-score: %.4f" % (epoch, avg_loss, avg_accuracy, avg_f1))


def test(model, epoch):

    accuracies, f1_scores = [], []

    for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(testloader), disable=True):

        with torch.no_grad():
            inputs = inputs.to(device)
            targets = targets.to(device)

            inputs = inputs.float().permute(0, 3, 1, 2)
            inputs_agent = torch.nn.functional.interpolate(inputs.clone(), (args.LR_size, args.LR_size))

            # Run LR image through Policy Network
            logits = model.forward(inputs_agent, activation=False)

            preds = logits.type(torch.DoubleTensor).to(device)
            preds = torch.sigmoid(preds)
            preds[preds < 0.5] = 0.0
            preds[preds >= 0.5] = 1.0
            matches = (preds == targets).int()
            accuracy = matches.sum(dim=1).sum() / (matches.shape[0] * matches.shape[1])
            micro_f1 = f1_score(targets.cpu().numpy(), preds.cpu().numpy(), average='micro',
                                labels=np.unique(preds.cpu().numpy()), zero_division=1)

        accuracies.append(accuracy.item())
        f1_scores.append(micro_f1)

        avg_accuracy = sum(accuracies) / len(accuracies)
        avg_f1 = sum(f1_scores)/len(f1_scores)
        # log_value(f'test_loss', avg_loss, epoch)
        # log_value(f'test_accuracy', avg_accuracy, epoch)
        # log_value(f'test_F1_score', avg_f1, epoch)
        print(f"Test | accuracy: %.4f | F1-score: %.4f\n" % (avg_accuracy, avg_f1)) # dice coeff??

        state = {
            'agent': model.state_dict(),
            'epoch': epoch,
            'accuracy': avg_accuracy,
            'F1-score': avg_f1
        }
        torch.save(state, 'checkpoints/PN_pretrain_E_%d_F1_%.3f' % (epoch, avg_f1))

if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logdir = 'pretrainPN/logs'
    if not os.path.exists(logdir):
        os.system('mkdir ' + logdir)
    # Save the log values
    configure(logdir, flush_secs=5)

    if not os.path.exists(args.cv_dir):
        os.system('mkdir ' + args.cv_dir)

    # Load the images and targets
    trainset = LandsatDataset('data/train_agent85.pkl')
    testset = LandsatDataset('data/test_agent15.pkl')

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Load the policy network model
    model = utils.get_model(args.model)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 1

    print('--- Policy Network Pre-training ---')

    for epoch in range(start_epoch, start_epoch+args.max_epochs):
        train(model, epoch)
        if epoch % args.test_interval == 0:
            test(model, epoch)
