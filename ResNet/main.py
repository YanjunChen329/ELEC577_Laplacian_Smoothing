#------------------------------------------------------------------------------
# System module.
#------------------------------------------------------------------------------
import os
import random
import time
import copy
import argparse
import sys

#------------------------------------------------------------------------------
# Torch module.
#------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable

#------------------------------------------------------------------------------
# Numpy module.
#------------------------------------------------------------------------------
import numpy as np
import numpy.matlib

#------------------------------------------------------------------------------
# DNN module
#------------------------------------------------------------------------------
from resnet import *
# from vgg import *
from utils import *

from LS_SGD import *

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

if __name__ == '__main__':
    """
    The main function.
    """
    use_cuda = torch.cuda.is_available()
    global best_acc
    best_acc = 0
    start_epoch = 0
    
    #--------------------------------------------------------------------------
    # Load the Cifar10 data.
    #--------------------------------------------------------------------------
    print('==> Preparing data...')
    root = '../data_Cifar10'
    download = True
    
    normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    
    train_set = torchvision.datasets.CIFAR10(
        root=root,
        train=True,
        download=download,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    
    test_set = torchvision.datasets.CIFAR10(
        root=root,
        train=False,
        download=download,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    
    # Convert the data into appropriate torch format.
    kwargs = {'num_workers':1, 'pin_memory':True}
    
    batchsize_test = 128
    print('Batch size of the test set: ', batchsize_test)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=batchsize_test,
                                              shuffle=False, **kwargs)
    
    batchsize_train = 128
    print('Batch size of the train set: ', batchsize_train)
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=batchsize_train,
                                               shuffle=True, **kwargs)
    
    # batchsize_train = len(train_set)
    print('Total training (known) batch number: ', len(train_loader))
    print('Total testing batch number: ', len(test_loader))
    
    #--------------------------------------------------------------------------
    # Build the model
    #--------------------------------------------------------------------------
    # checkpoint = torch.load("./checkpoint_Cifar/ckpt.t7")
    # net = checkpoint["net"]
    # start_epoch = checkpoint["epoch"]

    # net = resnet20().cuda()
    #net = resnet32().cuda()
    #net = resnet44().cuda()
    #net = resnet56().cuda()
    
    # net = preact_resnet20().cuda()
    net = preact_resnet32().cuda()
    #net = preact_resnet44().cuda()
    # net = preact_resnet56().cuda()
    
    # Print the model's information
    # paramsList = list(net.parameters())
    # kk = 0
    # for ii in paramsList:
    #     l = 1
    #     print('The structure of this layer: ' + str(list(ii.size())))
    #     for jj in ii.size():
    #         l *= jj
    #     print('The number of parameters in this layer: ' + str(l))
    #     kk = kk+l
    # print('Total number of parameters: ' + str(kk))
    
    sigma = 0.5
    lr = 0.12  # Changed
    weight_decay = 5e-4
    
    # optimizer = Grad_SJO_SGD(net.parameters(), lr=lr, sigma=sigma, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)

    train_loss_arr = []
    train_acc_arr = []
    test_loss_arr = []
    test_acc_arr = []
    
    nepoch = 200
    for epoch in range(start_epoch, nepoch):
        print()
        print('Epoch ID: ', epoch)
        
        #----------------------------------------------------------------------
        # Training
        #----------------------------------------------------------------------
        if epoch >= 40 and (epoch//40 == epoch/40.0):
            lr = lr/10
            print("Descrease the Learning Rate, lr = ", lr)
            # optimizer = Grad_SJO_SGD(net.parameters(), lr=lr, sigma = sigma, momentum=0.9, weight_decay=weight_decay, nesterov=True)

        correct = 0; total = 0; train_loss = 0
        net.train()
        for batch_idx, (x, target) in enumerate(train_loader):
            optimizer.zero_grad()
            x, target = Variable(x.cuda()), Variable(target.cuda())
            score, loss = net(x, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.data
            _, predicted = torch.max(score.data, 1)
            total += target.size(0)
            correct += predicted.eq(target.data).cpu().sum()
            
            # progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            # % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
            sys.stdout.write('\rEpoch: %d | Batch: %d | Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (epoch, batch_idx, train_loss / (batch_idx + 1), 100. * float(correct) / total, correct,
                                total))
            sys.stdout.flush()

        train_loss_arr.append(train_loss / (batch_idx + 1))
        train_acc_arr.append(100. * float(correct) / total)
        
        #----------------------------------------------------------------------
        # Testing
        #----------------------------------------------------------------------
        test_loss = 0; correct = 0; total = 0
        net.eval()
        for batch_idx, (x, target) in enumerate(test_loader):
            x, target = Variable(x.cuda()), Variable(target.cuda())
            score, loss = net(x, target)
            
            test_loss += loss.data
            _, predicted = torch.max(score.data, 1)
            total += target.size(0)
            correct += predicted.eq(target.data).cpu().sum()

        acc = 100. * float(correct) / total
        print('\nTesting: Loss: %.3f | Acc: %.3f%% (%d/%d)'
              % (test_loss / len(test_loader), acc, correct, total))
        test_loss_arr.append(test_loss / len(test_loader))
        test_acc_arr.append(acc)

        #----------------------------------------------------------------------
        # Save the checkpoint
        #----------------------------------------------------------------------
        if acc > best_acc:
            print('Saving model...')
            state = {
                'net': net,
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint_Cifar'):
                os.mkdir('checkpoint_Cifar')
            torch.save(state, './checkpoint_Cifar/ckpt.t7')
            best_acc = acc

        np.savetxt("./checkpoint_Cifar/train_loss_presnet32.txt", train_loss_arr)
        np.savetxt("./checkpoint_Cifar/train_acc_presnet32.txt", train_acc_arr)
        np.savetxt("./checkpoint_Cifar/test_loss_presnet32.txt", test_loss_arr)
        np.savetxt("./checkpoint_Cifar/test_acc_presnet32.txt", test_acc_arr)

    print('The best acc: ', best_acc)
