from __future__ import print_function, absolute_import, division

import os
import time
import math
import datetime
import argparse
import os.path as path
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.soundmnist import SoundMNIST 
from dataset.sound import Sound
# from models.soundlenet5 import SoundLenet5
# from models.lenet5 import LeNet5
from models.snet import SNet, SNetPluse
from utils.misc import save_ckpt




def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch training script')
    # general parm
    parser.add_argument('-s', '--sound_root', default = '../data/sound_450/', type = str, help='data root' )
    parser.add_argument('--checkpoint', default='./save/sound/450/21', type=str, help='checkpoint directory')
    parser.add_argument('--snapshot', default=1, type=int, help='save models for every # epochs (default: 1)')

    # model related parm
    parser.add_argument('-b', '--batch_size', default = 4, type = int, help='batch size' )
    # to modify
    parser.add_argument('-e', '--epochs', default = 35, type = int, help='num of epoch' )
    # parser.add_argument('-e', '--epochs', default = 5, type = int, help='num of epoch' )
    parser.add_argument('--lr', default = 1e-3, type = float, help='initial learning rate' )
    parser.add_argument('--per_class_num', default = 21 , type = int, help='per_class_num' )
    parser.add_argument('--vis_device', default='0', type=str, help='set visiable device')

    args = parser.parse_args()

    return args

def main(args):
    cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = args.vis_device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = Sound(sound_root=args.sound_root, per_class_num=args.per_class_num, train=True)
    val_dataseet = Sound(sound_root=args.sound_root, per_class_num=args.per_class_num, train=False)

    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataseet, batch_size = args.batch_size, shuffle = False, num_workers=2, pin_memory=True)

    #creat model 
    print('==> model creating....')
    model = SNetPluse().to(device)
    print('==> mdoel has been created')
    print("==> Total parameters (reference): {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = 1e-4)
    # optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, momentum=0.80, weight_decay = 0, nesterov=True)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.5) 
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience = 2) 

    total_step = len(train_loader)
    start_epoch = 0

    best_val_acc =None
    
    glob_step = 0

    evl_step = 0

    ckpt_dir_path = path.join(args.checkpoint, datetime.datetime.now().isoformat())
    if not path.exists(ckpt_dir_path):
            os.makedirs(ckpt_dir_path)
            print('==> Making checkpoint dir: {}'.format(ckpt_dir_path))

    writer = SummaryWriter(log_dir=ckpt_dir_path)

    print('start training')
    for epoch in range(args.epochs):
        # Train for one epoch

        glob_step = train(train_loader, model, criterion, optimizer, device, writer, glob_step, total_step, epoch, args.epochs, scheduler)

        scheduler.step() # need only in step lr
        train_acc = eval(train_loader, model, device,  epoch , writer) 
        val_acc = eval(val_loader, model, device,  epoch , writer)
        torch.set_grad_enabled(True)
        print('Epoch:[{}/{}], Train ACC:{:.6f},Test Acc:{:.6f}' .format(epoch+1, args.epochs, train_acc, val_acc))


        # save best model
        # to modify
        if epoch + 1 > 10:
        # if epoch + 1 > 3:
              if best_val_acc is None or best_val_acc < val_acc:

                  best_val_acc = val_acc
                  train_acc = eval(train_loader, model, device,  epoch , writer)
                  torch.set_grad_enabled(True)
                  print('Epoch:[{}/{}], Train ACC:{:.4f}, Test ACC Score:{:.4f} ' .format(epoch+1, args.epochs, train_acc, best_val_acc))
                    
                  save_ckpt({'epoch': epoch + 1, 'lr': args.lr, 'step': glob_step, 'state_dict': model.state_dict(),
                                         'optimizer': optimizer.state_dict() }, ckpt_dir_path, epoch = epoch + 1, best_acc = best_val_acc, train_acc=train_acc)

    writer.close()
    

def train(train_loader, model, criterion, optimizer, device, writer, step, total_step, epoch, total_epoch, scheduler):
    ''' train one epoch'''

    torch.set_grad_enabled(True)
    tic = time.time()
    # thresholds = 0.6 
    for i, (sounds, labels) in enumerate(train_loader):
        step += 1 
        sounds = sounds.to(device)
        labels = labels.to(device)

        
        # Forward pass
        outputs,_ = model(sounds)
        
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.6f}' .format(epoch+1, total_epoch, i+1, total_step, loss.item()))

        writer.add_scalar('Training Loss', loss.item(), step)
    return step




def eval(test_loader, model, device, epoch, writer):
    # Switch to evaluate mode
    torch.set_grad_enabled(False)
    model.eval()

    correct = 0
    total = 0
    for i, (sounds,labels) in enumerate(test_loader):
        sounds = sounds.to(device)
        labels = labels.to(device)

        outputs,_ = model(sounds)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    test_acc = 100 * correct / total
    
    
    return test_acc





if __name__ == '__main__':
    main(parse_args())