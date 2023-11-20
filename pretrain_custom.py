"""
This file trains the "Policy Network" standalone with custom targets (saved as binary vectors) .

Train on different configurations:
    python pretrain_custom.py --model ResNet_Landsat8, ResNet18_Landsat8
       --lr 1e-4
       --cv_dir checkpoint directory
       --batch_size 1048 (Higher is better)
       --LR_size 8, 56 (Depends on the dataset)
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm
import torch.optim as optim
from utils import agent_utils
from utils.custom_dataloader import LandsatDataset
from tensorboard_logger import configure, log_value
from sklearn.metrics import f1_score
import numpy as np
import pickle
import argparse

parser = argparse.ArgumentParser(description='Network Supervised Training')
parser.add_argument('--num_samples', type=int, default=15)
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--model', default='ResNet')
parser.add_argument('--data_dir', default=f'experiments/pretrain_agent_custom/100/thres0.01/data/', help='data directory')
parser.add_argument('--cv_dir', default=f'experiments/pretrain_agent_custom/toy/', help='models and logs are saved here')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--max_epochs', type=int, default=20, help='total epochs to run')
parser.add_argument('--LR_size', type=int, default=32)
parser.add_argument('--test_interval', type=int, default=1, help='Every how many epoch to test the model')
parser.add_argument('--ckpt_interval', type=int, default=20, help='Every how many epoch to save the model')
args = parser.parse_args()

def save_stats(losses, f1_scores, stats_dict):

    avg_loss = sum(losses) / len(losses)
    # avg_accuracy = sum(accuracies) / len(accuracies)
    avg_f1 = sum(f1_scores) / len(f1_scores)
    stats_dict["loss"].append(avg_loss)
    # stats_dict["accuracy"].append(avg_accuracy)
    stats_dict["F-score"].append(avg_f1)

    return avg_loss, avg_f1

def label_wise_accuracy(pred, target):
    pred = (pred > 0.5).float()
    matches = (pred == target).float()
    label_accuracy = torch.mean(matches, dim=0)
    return label_accuracy

def train(model, epoch):

    losses, accuracies, f1_scores = [], [], []

    for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(trainloader), total=len(trainloader), disable=True):

        inputs = inputs.to(device)
        targets = targets.to(device)

        inputs = inputs.float().permute(0, 3, 1, 2)
        inputs_agent = torch.nn.functional.interpolate(inputs.clone(), (args.LR_size, args.LR_size))

        logits = model.forward(inputs_agent, activation=False)

        preds = logits.type(torch.DoubleTensor).to(device)

        loss = criterion(preds, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute metrics
        with torch.no_grad():
            preds = torch.sigmoid(preds)
            # acc = torch.mean(label_wise_accuracy(preds, targets)) # works same
            preds[preds < 0.5] = 0.0
            preds[preds >= 0.5] = 1.0
            matches = (preds == targets).int()
            accuracy = matches.sum(dim=1).sum() / (matches.shape[0] * matches.shape[1])

            weighted_f1 = f1_score(targets.cpu().numpy(), preds.cpu().numpy(), average='weighted',
                                labels=np.unique(preds.cpu().numpy()), zero_division=0)

        losses.append(loss.item())
        # accuracies.append(accuracy.item())
        f1_scores.append(weighted_f1)

    avg_loss, avg_f1 = save_stats(losses, f1_scores, train_stats)
    log_value(f'train_loss', avg_loss, epoch)
    print(f"Train: %d | loss: %.4f | F1-score: %.4f" % (epoch, avg_loss, avg_f1))

def test(model, epoch):

    losses, accuracies, f1_scores = [], [], []

    for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(testloader), disable=True):

        with torch.no_grad():
            inputs = inputs.to(device)
            targets = targets.to(device)

            inputs = inputs.float().permute(0, 3, 1, 2)
            inputs_agent = torch.nn.functional.interpolate(inputs.clone(), (args.LR_size, args.LR_size))

            # Run LR image through Policy Network
            logits = model.forward(inputs_agent, activation=False)

            preds = logits.type(torch.DoubleTensor).to(device)
            loss = criterion(preds, targets)
            preds = torch.sigmoid(preds)

            # acc = torch.mean(label_wise_accuracy(preds, targets))
            preds[preds < 0.5] = 0.0
            preds[preds >= 0.5] = 1.0
            matches = (preds == targets).int()
            # accuracy = matches.sum(dim=1).sum() / (matches.shape[0] * matches.shape[1])

            weighted_f1 = f1_score(targets.cpu().numpy(), preds.cpu().numpy(), average='weighted',
                                labels=np.unique(preds.cpu().numpy()), zero_division=0)

        losses.append(loss.item())
        # accuracies.append(accuracy.item())
        f1_scores.append(weighted_f1)

    avg_loss, avg_f1 = save_stats(losses, f1_scores, test_stats)
    log_value(f'val_loss', avg_loss, epoch)
    print(f"Test | loss: %.4f | F1-score: %.4f\n" % (avg_loss, avg_f1))

    if epoch % args.ckpt_interval == 0:
        state = {
            'agent': model.state_dict(),
            'epoch': epoch,
            'F1-score': avg_f1
        }
        torch.save(state, args.cv_dir+'/checkpoints/PN_pretrain_E_%d_F1_%.3f' % (epoch, avg_f1))


if __name__ == "__main__":

    if not os.path.exists(args.cv_dir+'/checkpoints'):
        os.makedirs(args.cv_dir+'/checkpoints')

    # Save the log values for tensorboard logger
    if not os.path.exists(args.cv_dir+'/logs'):
        os.makedirs(args.cv_dir+'/logs')
    configure(args.cv_dir+'/logs', flush_secs=5)

    agent_utils.save_args(__file__, args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the images and targets
    trainset = LandsatDataset(args.data_dir+'train.pkl')
    testset = LandsatDataset(args.data_dir+'test.pkl')

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Load the policy network model
    model = agent_utils.get_model(args.model)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_stats = {"loss": [], "F-score": []}
    test_stats = {"loss": [], "F-score": []}

    start_epoch = 1

    for epoch in range(start_epoch, args.max_epochs+1):
        train(model, epoch)
        if epoch % args.test_interval == 0:
            test(model, epoch)

    # Save the results
    with open(f"{args.cv_dir}train_stats_E{args.max_epochs}", "wb") as fp:
        pickle.dump(train_stats, fp)

    with open(f"{args.cv_dir}test_stats_E{args.max_epochs}", "wb") as fp:
        pickle.dump(test_stats, fp)
