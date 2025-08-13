import torch
from utils import progress_bar
from calculation import calculate_region
import numpy as np

def train(args, device, epoch, net, trainloader, criterion, optimizer, train_loss_list, train_accuracy_list):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    avg_train_loss = train_loss / len(trainloader)
    avg_train_accuracy = 100. * correct / total
    train_loss_list.append(avg_train_loss)
    train_accuracy_list.append(avg_train_accuracy) 


def test(device, net, criterion, testloader, test_loss_list, test_accuracy_list):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    avg_test_loss = test_loss / len(testloader)
    avg_test_accuracy = 100. * correct / total
    test_loss_list.append(avg_test_loss)
    test_accuracy_list.append(avg_test_accuracy)

def calculate_region_entropy(args, criterion, optimizer, scheduler, device, net, net_init, start_epoch, trainloader, testloader, trainset_no_random, testset):
    num_epochs = args.training_epochs
    train_loss_list = []
    train_accuracy_list = []
    test_loss_list = []
    test_accuracy_list = []
    samples_list = []
    for i in range(args.average_number):
        samples = []
        labels = []
        if args.data_choose == 1:
            random_indices = np.random.choice(len(trainset_no_random), 2, replace=False)
            for index in random_indices:
                sample, label = trainset_no_random[index]
                samples.append(sample)
                labels.append(label)
        elif args.data_choose == 2:
            random_index = np.random.choice(len(trainset_no_random))
            sample, label = trainset_no_random[random_index]
            random_direction_1 = torch.randn_like(sample)
            length = 0.1  
            new_point_1 = sample + length * random_direction_1
            samples.append(sample)
            samples.append(new_point_1)
        elif args.data_choose == 3:
            random_indices = np.random.choice(len(trainset_no_random), 3, replace=False)
            for index in random_indices:
                sample, label = trainset_no_random[index]
                samples.append(sample)
                labels.append(label)
        elif args.data_choose == 4:
            random_indices = np.random.choice(len(trainset_no_random), 4, replace=False)
            for index in random_indices:
                sample, label = trainset_no_random[index]
                samples.append(sample)
                labels.append(label)
        elif args.data_choose == 5:
            random_indices = np.random.choice(len(trainset_no_random), 5, replace=False)
            for index in random_indices:
                sample, label = trainset_no_random[index]
                samples.append(sample)
                labels.append(label)
        elif args.data_choose == 6:
            random_indices = np.random.choice(len(trainset_no_random), 6, replace=False)
            for index in random_indices:
                sample, label = trainset_no_random[index]
                samples.append(sample)
                labels.append(label)
      
        samples_list.append(samples)
        
    average_region_list = []
    for epoch in range(start_epoch, start_epoch + num_epochs + 1):
        if epoch % args.skip_plot == 0:
            regions_list = []
            calculate_region(args, regions_list, device, net, samples_list)
            average_region = np.mean(regions_list)
            average_region_list.append(average_region)
        train(args, device, epoch, net, trainloader, criterion, optimizer, train_loss_list, train_accuracy_list)
        test(device, net, criterion, testloader, test_loss_list, test_accuracy_list)
        if scheduler is not None:
            scheduler.step()
   
    f = open('results/' + str(args.net) + '_' + args.dataset + '.txt', 'a')
    f.write(str(args.data_choose) + ' ' + ' '.join([str(elem) for elem in average_region_list]) + ' ' + str(train_accuracy_list[-1]) + ' ' + str(test_accuracy_list[-1]) + ' ' + str(args.weight_decay) + ' ' + str(args.lr) + ' ' + str(args.batch_size) + '\n')
        