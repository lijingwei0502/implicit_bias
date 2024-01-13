import torch
import os
from utils import progress_bar
from plot import plot_loss_accuracy, plot_one_decision_boundary, calculate_region
import matplotlib.pyplot as plt
from calculate import epochcoincide
import numpy as np


def mixup_data(x, y, alpha=1.0, device='cuda:2'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)  # 确保index在正确的设备上

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train(args, device, epoch, net, trainloader, criterion, optimizer, train_loss_list, train_accuracy_list):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        # add mixup
        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, args.mixup, device)
        optimizer.zero_grad()
        outputs = net(inputs)
        #loss = criterion(outputs, targets)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
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

def train_test(args, criterion, optimizer, scheduler, device, net, start_epoch, trainloader, testloader):
    dir_name = args.dir + '/'
    num_epochs = args.training_epochs
    train_loss_list = []
    train_accuracy_list = []
    test_loss_list = []
    test_accuracy_list = []
    
    for epoch in range(start_epoch, start_epoch + num_epochs):  
        train(args, device, epoch, net, trainloader, criterion, optimizer, train_loss_list, train_accuracy_list)
        test(device, net, criterion, testloader, test_loss_list, test_accuracy_list)
        scheduler.step()
            
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'epoch': epoch,
        }
        torch.save(state, dir_name + 'checkpoint.pth')
        
    plot_loss_accuracy(dir_name, start_epoch, num_epochs, train_loss_list, test_loss_list, train_accuracy_list, test_accuracy_list)


def small_plot(args, criterion, optimizer, scheduler, device, net, start_epoch, trainloader, testloader, sample_1, sample_2, sample_3, close_x, close_y, close_label):
    dir_name = args.dir + '/'
    num_epochs = args.training_epochs
    train_loss_list = []
    train_accuracy_list = []
    test_loss_list = []
    test_accuracy_list = []
    epochs_per_figure = 1  # 每个大图里面有 1 个小图
    num_figures = num_epochs // epochs_per_figure
    regions_list = []
    for fig_idx in range(num_figures):
        for local_epoch in range(epochs_per_figure):
            epoch = start_epoch + fig_idx * epochs_per_figure + local_epoch  # 计算全局的epoch数
            train(args, device, epoch, net, trainloader, criterion, optimizer, train_loss_list, train_accuracy_list)
            test(device, net, criterion, testloader, test_loss_list, test_accuracy_list)
            scheduler.step()
            plot_one_decision_boundary(regions_list, device, net, epoch, sample_1, sample_2, sample_3, close_x, close_y, close_label)
            
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'epoch': epoch,
        }
        torch.save(state, dir_name + 'checkpoint.pth')
        plt.tight_layout()
        plt.savefig(dir_name + f'boundaries_{fig_idx + 1}.png')  # 保存每个
        # 将matplotlib图像保存为数组
    
    plt.figure()
    plot_loss_accuracy(dir_name, start_epoch, num_epochs, regions_list, train_loss_list, test_loss_list, train_accuracy_list, test_accuracy_list)

def calculate_region_entropy(args, criterion, optimizer, scheduler, device, net, start_epoch, trainloader, testloader, trainset_no_random, testset):
    num_epochs = args.training_epochs
    train_loss_list = []
    train_accuracy_list = []
    test_loss_list = []
    test_accuracy_list = []
    samples_list = []
    for i in range(args.average_number):
        samples = []
        labels = []
        if args.data_choose == 0:
            random_indices = np.random.choice(len(trainset_no_random), 3, replace=False)
            for index in random_indices:
                sample, label = trainset_no_random[index]
                samples.append(sample)
                labels.append(label)
        elif args.data_choose == 1:
            random_indices = np.random.choice(len(testset), 3, replace=False)
            for index in random_indices:
                sample, label = testset[index]
                samples.append(sample)
                labels.append(label)
        elif args.data_choose == 2:
            # 从测试集中随机选择一个样本
            random_index = np.random.choice(len(testset))
            sample, label = testset[random_index]

            # 生成两个高斯随机向量
            random_direction_1 = torch.randn_like(sample)
            random_direction_2 = torch.randn_like(sample)

            # 设置延伸长度
            length = 0.1  # 可以调整这个长度

            # 生成两个新的点
            new_point_1 = sample + length * random_direction_1
            new_point_2 = sample + length * random_direction_2
            samples.append(sample)
            samples.append(new_point_1)
            samples.append(new_point_2)
        elif args.data_choose == 3:
            # 生成第一个点（空间内的随机点）
            random_index = np.random.choice(len(testset))
            sample, label = testset[random_index]

            # 添加小的高斯扰动
            noise_level = 0.05  # 可以调整这个值
            noise = torch.randn_like(sample) * noise_level

            # 生成随机点
            random_point = sample + noise

            # 生成两个高斯随机向量
            random_direction_1 = torch.randn_like(random_point)
            random_direction_2 = torch.randn_like(random_point)

            # 设置延伸长度
            length = 0.1  # 可以调整这个长度

            # 生成两个新的点
            new_point_1 = random_point + length * random_direction_1
            new_point_2 = random_point + length * random_direction_2
            samples.append(random_point)
            samples.append(new_point_1)
            samples.append(new_point_2)
        elif args.data_choose == 4:
            random_indices = np.random.choice(len(testset), 2, replace=False)
            for index in random_indices:
                sample, label = testset[index]
                samples.append(sample)
                labels.append(label)
        elif args.data_choose == 5:
            random_indices = np.random.choice(len(trainset_no_random), 2, replace=False)
            for index in random_indices:
                sample, label = trainset_no_random[index]
                samples.append(sample)
                labels.append(label)
        elif args.data_choose == 6:
            # 从测试集中随机选择一个样本
            random_index = np.random.choice(len(testset))
            sample, label = testset[random_index]

            # 生成两个高斯随机向量
            random_direction_1 = torch.randn_like(sample)
            
            # 设置延伸长度
            length = 0.1  # 可以调整这个长度

            # 生成两个新的点
            new_point_1 = sample + length * random_direction_1
            samples.append(sample)
            samples.append(new_point_1)
        elif args.data_choose == 7:
            # 生成第一个点（空间内的随机点）
            random_index = np.random.choice(len(testset))
            sample, label = testset[random_index]

            # 添加小的高斯扰动
            noise_level = 0.05
            noise = torch.randn_like(sample) * noise_level
            random_point = sample + noise
            random_direction_1 = torch.randn_like(sample)
            # 设置延伸长度
            length = 0.1
            # 生成两个新的点
            new_point_1 = random_point + length * random_direction_1
            samples.append(random_point)
            samples.append(new_point_1)
        if args.different_three:
            if labels[0] == labels[1] or labels[1] == labels[2] or labels[0] == labels[2]:
                i -= 1
                continue
        samples_list.append(samples)

    average_region_list = []
    variance_region_list = []
    average_entropy_list = []
    variance_entropy_list = []
    for epoch in range(start_epoch, start_epoch + num_epochs + 1):
        scheduler.step()
        if epoch % args.skip_plot == 0:
            regions_list = []
            entropy_list = []
            calculate_region(args, epoch, regions_list, entropy_list, device, net, samples_list)
            average_region = np.mean(regions_list)
            variance_region = np.var(regions_list)
            average_region_list.append(average_region)
            variance_region_list.append(variance_region)
            average_entropy = np.mean(entropy_list)
            average_entropy_list.append(average_entropy)
            variance_entropy = np.var(entropy_list)
            variance_entropy_list.append(variance_entropy)
        train(args, device, epoch, net, trainloader, criterion, optimizer, train_loss_list, train_accuracy_list)
        test(device, net, criterion, testloader, test_loss_list, test_accuracy_list)

              
    plot_loss_accuracy(args, start_epoch, num_epochs, average_region_list, average_entropy_list, variance_region_list, variance_entropy_list, train_loss_list, test_loss_list, train_accuracy_list, test_accuracy_list)
    f = open('final/' + str(args.net) + '.txt', 'a')
    f.write(' '.join([str(elem) for elem in average_region_list]) + ' ' + str(train_accuracy_list[-1]) + ' ' + str(test_accuracy_list[-1]) + ' ' + str(args.weight_decay) + ' ' + str(args.lr) + ' ' + str(args.batch_size) + '\n')
    # g = open('final/entropy' + str(args.device) + '.txt', 'a')
    # g.write(' '.join([str(elem) for elem in average_entropy_list]) + ' ' + str(test_accuracy_list[-1]) + ' ' + str(args.weight_decay) + ' ' + str(args.lr) + ' ' + str(args.batch_size) + ' ' + str(args.net) + '\n')
    
