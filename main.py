import os
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
from net import get_net
from net_imagenet import get_net_imagenet

from load_dataset import load_train_test
from train_and_test import calculate_region_entropy
import pynvml

def found_device():
    default_device=0
    default_memory_threshold=500
    pynvml.nvmlInit()
    while True:
        handle=pynvml.nvmlDeviceGetHandleByIndex(default_device)
        meminfo=pynvml.nvmlDeviceGetMemoryInfo(handle)
        used=meminfo.used/1024**2
        if used<default_memory_threshold:
            break
        else:
            default_device+=1
        if default_device>=8:
            default_device=0
            default_memory_threshold+=1000
    pynvml.nvmlShutdown()
    return str(default_device)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--resume', default=False, type=bool, help='resume from checkpoint') # 是否从checkpoint开始训练
    parser.add_argument('--dir', default='results', type=str, help='directory to save figs')
    parser.add_argument('--seed', default='0', type=int, help='random seed')
    parser.add_argument('--plot', default=False, type=bool, help='plot decision boundary')
    parser.add_argument('--training_epochs', default=200, type=int, help='training epochs')
    parser.add_argument('--mixup', default=0, type=int, help='whether use mixup')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--net', default='Resnet18', type=str, help='network')
    parser.add_argument('--scope_l', default=0, type=int, help='plane_scope_l')
    parser.add_argument('--scope_r', default=1, type=int, help='plane_scope_r')
    parser.add_argument('--skip_plot',default=1,type=int, help= 'how to plot')
    parser.add_argument('--different_three',default=0,type=int, help= 'whether choose different three points')
    parser.add_argument('--dataset',default='cifar10',type=str, help= 'dataset')
    parser.add_argument('--optimizer', default='sgd', type=str, help='optimizer')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--scheduler', default='cosine', type=str, help='scheduler')
    parser.add_argument('--average_number', default=100, type=int, help='average')
    parser.add_argument('--random_crop', default=32,type=int,help='random crop')
    parser.add_argument('--random_horizontal_flip', default=0.5,type=float,help='random horizontal flip')
    parser.add_argument('--device', default=0,type=int,help='use which device')
    parser.add_argument('--batch_size', default=256,type=int,help='batch size')
    parser.add_argument('--random', default=0,type=int,help='whether_random_crop')
    parser.add_argument('--data_choose', default=0,type=int,help='how to choose data')
    parser.add_argument('--task', default='correlation',type=str,help='task')
    args = parser.parse_args()
    #for debug
    # args.dataset = "imagenet-1k"
    set_seed(args.seed)
    device = 'cuda:' + found_device() if torch.cuda.is_available() else 'cpu'
    start_epoch = 0
    trainset_no_random, testset, trainloader, testloader = load_train_test(args)
    if args.dataset == 'cifar10':
        net, criterion, optimizer, scheduler = get_net(args, device)
    else:
        net, criterion, optimizer, scheduler = get_net_imagenet(args, device)
    if not os.path.exists(args.dir):
        os.makedirs(args.dir)
    if not os.path.exists(args.task):
        os.makedirs(args.task)
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(args.dir + '/checkpoint.pth')
        net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch'] + 1
    
    calculate_region_entropy(args, criterion, optimizer, scheduler, device, net, start_epoch, trainloader, testloader, trainset_no_random, testset)
