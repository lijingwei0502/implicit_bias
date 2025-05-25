from models import *
import torch.backends.cudnn as cudnn
import torch.optim as optim


def get_net(args, device):
    num_classes = 10
    if args.dataset == "imagenet-1k":
        num_classes = 1000
    print('[dataset]: ', args.dataset, '[num_classes]: ', num_classes, '[model]: ', args.net)

    if args.net == 'Resnet18':
        net = ResNet18(num_classes = num_classes)
    elif args.net == 'Resnet34':
        net = ResNet34()
    elif args.net == 'VGG19':
        net = VGG('VGG19')
    elif args.net == 'GoogLeNet':
        net = GoogLeNet()
    elif args.net == 'DenseNet121':
        net = DenseNet121()
    elif args.net == 'ResNeXt29_2x64d':
        net = ResNeXt29_2x64d()
    elif args.net == 'MobileNet':
        net = MobileNet()
    elif args.net == 'MobileNetV2':
        net = MobileNetV2()
    elif args.net == 'DPN92':
        net = DPN92()
    elif args.net == 'SENet18':
        net = SENet18()
    elif args.net == 'ShuffleNetV2':
        net = ShuffleNetV2(1)
    elif args.net == 'EfficientNetB0':
        net = EfficientNetB0()
    elif args.net == 'RegNetX_200MF':
        net = RegNetX_200MF()
    elif args.net == 'SimpleDLA':
        net = SimpleDLA()
    
    
    # cifar100 change output layer
    if args.dataset == 'cifar100':
        net.linear = nn.Linear(512, 100)
        net.linear.weight.data.normal_(0, 0.01)
        net.linear.bias.data.zero_()

    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay = args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay = args.weight_decay)
    elif args.optimizer == 'adagrad':
        optimizer = optim.Adagrad(net.parameters(), lr=args.lr, weight_decay = args.weight_decay)
    elif args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(net.parameters(), lr=args.lr, weight_decay = args.weight_decay)
    elif args.optimizer == 'adadelta':
        optimizer = optim.Adadelta(net.parameters(), lr=args.lr, weight_decay = args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay = args.weight_decay)
    
    if args.scheduler == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.1)
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.training_epochs)
    elif args.scheduler == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    elif args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    elif args.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)
    elif args.scheduler == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1)
    elif args.scheduler == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=len(trainloader), epochs=args.training_epochs)
    elif args.scheduler == 'cosineannealingwarmrestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    elif args.scheduler == 'none': # change
        scheduler = None

    return net, criterion, optimizer, scheduler

    