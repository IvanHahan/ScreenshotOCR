from tqdm import tqdm
import argparse
import os
import random
import shutil
import time
import warnings
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from ocr.efficientdet import EfficientDet
from ocr.efficientdet.losses import FocalLoss
from ocr.data.letter_dataset import LetterDataset

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--image_dir', default='/Users/UnicornKing/PyCharmProjects/ScreenshotOCR/data/images/raw')
parser.add_argument('--annot_dir', default='/Users/UnicornKing/PyCharmProjects/ScreenshotOCR/data/annotations')
parser.add_argument('--network', default='efficientdet-d0')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--num_epoch', default=5, type=int,
                    help='Num epoch for training')
parser.add_argument('--batch_size', default=1, type=int,
                    help='Batch size for training')
parser.add_argument('--num_class', default=1, type=int,
                    help='Number of class used in model')
parser.add_argument('--device', default=[0, 1], type=list,
                    help='Use CUDA to train model')
parser.add_argument('--grad_accumulation_steps', default=1, type=int,
                    help='Number of gradient accumulation steps')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--save_folder', default='./saved/weights/', type=str,
                    help='Directory for saving checkpoint models')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--dist-url', default='env://', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=24, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

iteration = 1


def train(train_loader, model, scheduler, optimizer, epoch, args):
    global iteration
    print("{} epoch: \t start training....".format(epoch))
    start = time.time()
    total_loss = []
    model.train()
    model.is_training = True
    model.freeze_bn()
    optimizer.zero_grad()
    for idx, (images, annotations) in enumerate(train_loader):
        images = images.float()
        annotations = annotations
        classification_loss, regression_loss = model([images, annotations])
        classification_loss = classification_loss.mean()
        regression_loss = regression_loss.mean()
        loss = classification_loss + regression_loss
        if bool(loss == 0):
            print('loss equal zero(0)')
            continue
        loss.backward()
        if (idx + 1) % args.grad_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            optimizer.zero_grad()

        total_loss.append(loss.item())
        if(iteration % 300 == 0):
            print('{} iteration: training ...'.format(iteration))
            ans = {
                'epoch': epoch,
                'iteration': iteration,
                'cls_loss': classification_loss.item(),
                'reg_loss': regression_loss.item(),
                'mean_loss': np.mean(total_loss)
            }
            for key, value in ans.items():
                print('    {:15s}: {}'.format(str(key), value))
        iteration += 1
    scheduler.step(np.mean(total_loss))
    result = {
        'time': time.time() - start,
        'loss': np.mean(total_loss)
    }
    for key, value in result.items():
        print('    {:15s}: {}'.format(str(key), value))


def test(dataset, model, epoch, args):
    print("{} epoch: \t start validation....".format(epoch))
    model = model.module
    model.eval()
    model.is_training = False


def main_worker(gpu, ngpus_per_node, args):
    device = args.device

    # Training dataset
    train_dataset = LetterDataset(args.image_dir, args.annot_dir)

    train_loader = torch.utils.data.DataLoader(train_dataset, 1, shuffle=False)

    checkpoint = []
    if(args.resume is not None):
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
        params = checkpoint['parser']
        args.num_class = params.num_class
        args.network = params.network
        args.start_epoch = checkpoint['epoch'] + 1
        del params

    model = EfficientDet(num_classes=1,
                         network=args.network
                         )
    if(args.resume is not None):
        model.load_state_dict(checkpoint['state_dict'])
    del checkpoint
    # define loss function (criterion) , optimizer, scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, verbose=True)
    cudnn.benchmark = True

    for epoch in range(args.start_epoch, args.num_epoch):
        train(train_loader, model, scheduler, optimizer, epoch, args)


        state = {
            'epoch': epoch,
            'parser': args,
            'state_dict': model.state_dict()
        }

        torch.save(
            state,
            os.path.join(
                args.save_folder,
                args.network,
                "checkpoint_{}.pth".format(epoch)))


def main():
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    main_worker(args.gpu, 0, args)


if __name__ == "__main__":
    main()
