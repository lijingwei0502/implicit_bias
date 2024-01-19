import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import Subset
import numpy as np

def load_train_test(args):
    if args.random:
        transform_train_cifar10 = transforms.Compose([
            transforms.RandomCrop(args.random_crop, padding=4),
            transforms.RandomHorizontalFlip(args.random_horizontal_flip),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_train_imagenet_1k = transforms.Compose([
            transforms.Resize(size=(224, 224), antialias=True),
            transforms.RandomResizedCrop(224),		
            transforms.RandomHorizontalFlip(),		
            transforms.ToTensor(),				
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))	
        ])

    else:
        transform_train_cifar10 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_train_imagenet_1k = transforms.Compose([
            transforms.Resize(size=(224, 224), antialias=True),
            transforms.ToTensor(),				
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))	
        ])


    transform_train_cifar10_no_random = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


    transform_test_cifar10 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


    transform_train_cifar100 = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    transform_train_cifar100_no_random = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    transform_test_cifar100 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    transform_train_svhn = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
    ])

    transform_train_svhn_no_random = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
    ])

    transform_test_svhn = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
    ])

    transform_test_imagenet_1k = transforms.Compose([
        transforms.Resize(size=(224, 224), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    transform_train_imagenet_1k_no_random = transforms.Compose([
        transforms.Resize(size=(224, 224), antialias=True),
        transforms.ToTensor(),				
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))	
    ])

    class RemappedSubset(torch.utils.data.Dataset):
        def __init__(self, dataset, indices, label_map):
            self.dataset = dataset
            self.indices = indices
            self.label_map = label_map

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            original_tuple = self.dataset[self.indices[idx]]
            return (original_tuple[0], self.label_map[original_tuple[1]])


    if args.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(
            root='data', train=True, download=True, transform=transform_train_cifar10)
        trainset_no_random = torchvision.datasets.CIFAR10(
            root='data', train=True, download=True, transform=transform_train_cifar10_no_random)
        testset = torchvision.datasets.CIFAR10(
            root='data', train=False, download=True, transform=transform_test_cifar10)
        
    elif args.dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(
            root='data', train=True, download=True, transform=transform_train_cifar100)
        trainset_no_random = torchvision.datasets.CIFAR100( 
            root='data', train=True, download=True, transform=transform_train_cifar100_no_random)
        testset = torchvision.datasets.CIFAR100(
            root='data', train=False, download=True, transform=transform_test_cifar100)
        
    # use svhn
    elif args.dataset == 'svhn':
        trainset = torchvision.datasets.SVHN(
            root='data', split='train', download=True, transform=transform_train_svhn)
        trainset_no_random = torchvision.datasets.SVHN(
            root='data', split='train', download=True, transform=transform_train_svhn_no_random)
        testset = torchvision.datasets.SVHN(
            root='data', split='test', download=True, transform=transform_test_svhn)

    elif args.dataset == 'imagenet-1k':
        trainset = datasets.ImageFolder('data/imagenet-1k/train', transform=transform_train_imagenet_1k)
        testset = datasets.ImageFolder('data/imagenet-1k/val', transform=transform_test_imagenet_1k)
        trainset_no_random = datasets.ImageFolder('data/imagenet-1k/train', transform=transform_train_imagenet_1k_no_random)
        # testset_no_random = datasets.ImageFolder('data/imagenet-1k/val', transform=transform_test_imagenet_1k)
        

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    return trainset_no_random, testset, trainloader, testloader

