import torch
import os
from utils import progress_bar
from plot import plot_loss_accuracy, calculate_region, calculate_machine_region
import matplotlib.pyplot as plt
import numpy as np
from measure import get_all_measures


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

def compute_full_gradient(net, trainloader, criterion, device):
    """计算全批量梯度"""
    net.zero_grad()
    full_grad = [torch.zeros_like(p) for p in net.parameters()]
    
    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        with torch.no_grad():
            for param, fg in zip(net.parameters(), full_grad):
                fg += param.grad / len(trainloader)
    
    return full_grad

def calculate_variance(net, trainloader, criterion, device):
    gradients = []
    # 在训练开始前计算全批量梯度
    full_gradient = compute_full_gradient(net, trainloader, criterion, device)
    
    for inputs, targets in trainloader:
        net.zero_grad()
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        # 保存每个参数的梯度
        batch_grad = [param.grad.clone() for param in net.parameters()]
        gradients.append(batch_grad)

    # 计算梯度方差
    variances = []
    for bg in gradients:
        variance = sum(torch.sum((g - fg) ** 2) for g, fg in zip(bg, full_gradient))
        variances.append(variance)

    mean_variance = sum(variances) / len(variances)
    return mean_variance

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
            # 从测试集中随机选择一个样本
            random_index = np.random.choice(len(trainset_no_random))
            sample, label = trainset_no_random[random_index]
            # 生成两个高斯随机向量
            random_direction_1 = torch.randn_like(sample)
            # 设置延伸长度
            length = 0.1  # 可以调整这个长度
            # 生成两个新的点
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
        elif args.data_choose == 7:
            # 从训练集中随机选择两个样本，然后在这两个样本之间随机选择两个他们的线性组合，作为samples
            random_indices = np.random.choice(len(trainset_no_random), 2, replace=False)
            sample_1, label_1 = trainset_no_random[random_indices[0]]
            sample_2, label_2 = trainset_no_random[random_indices[1]]
            # 生成两个随机系数
            alpha = np.random.rand()
            beta = np.random.rand()
            # 生成两个新的点
            new_point_1 = alpha * sample_1 + (1 - alpha) * sample_2
            new_point_2 = beta * sample_1 + (1 - beta) * sample_2
            samples.append(new_point_1)
            samples.append(new_point_2)
            
        samples_list.append(samples)
        
    
    average_region_list = []
    variance_region_list = []
    average_entropy_list = []
    variance_entropy_list = []
    if args.machine_learning > 0:
        if args.machine_learning == 1:
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.metrics import accuracy_score
            # 初始化KNN分类器，这里K设为3
            knn = KNeighborsClassifier(n_neighbors=args.k, p = args.p, weights=args.weights)
            
            xtrain = []
            ytrain = []
            for i in range(len(trainset_no_random)):
                x, y = trainset_no_random[i]
                x = x.reshape(-1)
                xtrain.append(x)
                ytrain.append(y)

            xtrain = np.array(xtrain)
            ytrain = np.array(ytrain)
            Xtest = []
            Ytest = []
            for i in range(len(testset)):
                x, y = testset[i]
                x = x.reshape(-1)
                Xtest.append(x)
                Ytest.append(y)
            Xtest = np.array(Xtest)
            Ytest = np.array(Ytest)
            # 使用训练数据拟合模型
            knn.fit(xtrain, ytrain)
            # 打印训练准确率
            tr_acc = knn.score(xtrain, ytrain)
            print(tr_acc)
            # 在测试集上进行预测
            Ypred = knn.predict(Xtest)

            # 计算准确率
            accuracy = accuracy_score(Ytest, Ypred)
            print(accuracy)
            regions_list = []
            entropy_list = []
            calculate_machine_region(args, regions_list, entropy_list, device, knn, samples_list)
            average_region = np.mean(regions_list)
            f = open('knn.txt', 'a')
            f.write(str(average_region) + ' ' + str(tr_acc) + ' ' + str(accuracy) + '\n')
        elif args.machine_learning == 2:
            from sklearn.svm import SVC
            from sklearn.metrics import accuracy_score
            # 初始化SVM分类器
            svm = SVC(kernel = args.kernel, C = args.C)
            
            xtrain = []
            ytrain = []
            for i in range(len(trainset_no_random)):
                x, y = trainset_no_random[i]
                x = x.reshape(-1)
                xtrain.append(x)
                ytrain.append(y)

            xtrain = np.array(xtrain)
            ytrain = np.array(ytrain)
            Xtest = []
            Ytest = []
            for i in range(len(testset)):
                x, y = testset[i]
                x = x.reshape(-1)
                Xtest.append(x)
                Ytest.append(y)
            Xtest = np.array(Xtest)
            Ytest = np.array(Ytest)
            # 使用训练数据拟合模型
            svm.fit(xtrain, ytrain)
            # # 打印训练准确率
            tr_acc = svm.score(xtrain, ytrain)
            print(tr_acc)
            # 在测试集上进行预测
            Ypred = svm.predict(Xtest)

            # 计算准确率
            accuracy = accuracy_score(Ytest, Ypred)
            print(accuracy)
            regions_list = []
            entropy_list = []
            for sample_1, sample_2 in samples_list:
                print(sample_1)
                print(sample_2)
            calculate_machine_region(args, regions_list, entropy_list, device, svm, samples_list)
            average_region = np.mean(regions_list)
            f = open('svm.txt', 'a')
            f.write(str(average_region) + ' ' + str(tr_acc) + ' ' + str(accuracy) + '\n')
        elif args.machine_learning == 3:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score
            # 初始化随机森林分类器
            rf = RandomForestClassifier(max_depth=args.max_depth, criterion=args.criterion, n_estimators = args.n_estimators)
            
            xtrain = []
            ytrain = []
            for i in range(len(trainset_no_random)):
                x, y = trainset_no_random[i]
                x = x.reshape(-1)
                xtrain.append(x)
                ytrain.append(y)

            xtrain = np.array(xtrain)
            ytrain = np.array(ytrain)
            Xtest = []
            Ytest = []
            for i in range(len(testset)):
                x, y = testset[i]
                x = x.reshape(-1)
                Xtest.append(x)
                Ytest.append(y)
            Xtest = np.array(Xtest)
            Ytest = np.array(Ytest)
            # 使用训练数据拟合模型
            rf.fit(xtrain, ytrain)
            # 打印训练准确率
            tr_acc = rf.score(xtrain, ytrain)
            print(tr_acc)
            # 在测试集上进行预测
            Ypred = rf.predict(Xtest)

            # 计算准确率
            accuracy = accuracy_score(Ytest, Ypred)
            print(accuracy)
            regions_list = []
            entropy_list = []
            calculate_machine_region(args, regions_list, entropy_list, device, rf, samples_list)
            average_region = np.mean(regions_list)
            f = open('rf.txt', 'a')
            f.write(str(average_region) + ' ' + str(tr_acc) + ' ' + str(accuracy) + '\n')
        elif args.machine_learning == 4:
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score
            # 初始化逻辑回归分类器
            lr = LogisticRegression(penalty = args.penalty, C = args.C)
            
            xtrain = []
            ytrain = []
            for i in range(len(trainset_no_random)):
                x, y = trainset_no_random[i]
                x = x.reshape(-1)
                xtrain.append(x)
                ytrain.append(y)

            xtrain = np.array(xtrain)
            ytrain = np.array(ytrain)
            Xtest = []
            Ytest = []
            for i in range(len(testset)):
                x, y = testset[i]
                x = x.reshape(-1)
                Xtest.append(x)
                Ytest.append(y)
            Xtest = np.array(Xtest)
            Ytest = np.array(Ytest)
            # 使用训练数据拟合模型
            lr.fit(xtrain, ytrain)
            # 打印训练准确率
            tr_acc = lr.score(xtrain, ytrain)
            print(tr_acc)
            # 在测试集上进行预测
            Ypred = lr.predict(Xtest)

            # 计算准确率
            accuracy = accuracy_score(Ytest, Ypred)
            print(accuracy)
            regions_list = []
            entropy_list = []
            calculate_machine_region(args, regions_list, entropy_list, device, lr, samples_list)
            average_region = np.mean(regions_list)
            f = open('lr.txt', 'a')
            f.write(str(average_region) + ' ' + str(tr_acc) + ' ' + str(accuracy) + '\n')
        # use decision tree
        elif args.machine_learning == 5:
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.metrics import accuracy_score
            # 初始化决策树分类器
            dt = DecisionTreeClassifier(max_depth=args.max_depth, criterion=args.criterion, splitter=args.splitter)
            
            xtrain = []
            ytrain = []
            for i in range(len(trainset_no_random)):
                x, y = trainset_no_random[i]
                x = x.reshape(-1)
                xtrain.append(x)
                ytrain.append(y)

            xtrain = np.array(xtrain)
            ytrain = np.array(ytrain)
            Xtest = []
            Ytest = []
            for i in range(len(testset)):
                x, y = testset[i]
                x = x.reshape(-1)
                Xtest.append(x)
                Ytest.append(y)
            Xtest = np.array(Xtest)
            Ytest = np.array(Ytest)
            # 使用训练数据拟合模型
            dt.fit(xtrain, ytrain)
            # 打印训练准确率
            tr_acc = dt.score(xtrain, ytrain)
            print(tr_acc)
            # 在测试集上进行预测
            Ypred = dt.predict(Xtest)

            # 计算准确率
            accuracy = accuracy_score(Ytest, Ypred)
            print(accuracy)
            regions_list = []
            entropy_list = []
            calculate_machine_region(args, regions_list, entropy_list, device, dt, samples_list)
            average_region = np.mean(regions_list)
            f = open('dt.txt', 'a')
            f.write(str(average_region) + ' ' + str(tr_acc) + ' ' + str(accuracy) + '\n')

    else:
        for epoch in range(start_epoch, start_epoch + num_epochs + 1):
            if epoch % args.skip_plot == 0:
                regions_list = []
                entropy_list = []
                calculate_region(args, epoch, regions_list, entropy_list, device, net, samples_list)
                average_region = np.mean(regions_list)
                variance_region = np.var(regions_list)
                print('average region: ', average_region)
                average_region_list.append(average_region)
                variance_region_list.append(variance_region)
                average_entropy = np.mean(entropy_list)
                average_entropy_list.append(average_entropy)
                variance_entropy = np.var(entropy_list)
                variance_entropy_list.append(variance_entropy)
                train(args, device, epoch, net, trainloader, criterion, optimizer, train_loss_list, train_accuracy_list)
                test(device, net, criterion, testloader, test_loss_list, test_accuracy_list)
                if scheduler is not None:
                    scheduler.step()
        plot_loss_accuracy(args, start_epoch, num_epochs, average_region_list, average_entropy_list, variance_region_list, variance_entropy_list, train_loss_list, test_loss_list, train_accuracy_list, test_accuracy_list)
        if args.measure == 1:
            net.eval()
            net_init.eval()
            measure = get_all_measures(net, net_init, trainloader)
            res_dict = {str(key): value for key, value in measure.items()}
            q = open('measure.txt', 'a')
            q.write(
    str(res_dict['ComplexityType.FRO_DIST']) + ' ' +
    str(res_dict['ComplexityType.MARGIN']) + ' ' +
    str(res_dict['ComplexityType.MARGIN'] / res_dict['ComplexityType.FRO_DIST']) + ' ' +
    str(res_dict['ComplexityType.LOG_PROD_OF_SPEC']) + ' ' +  # Spectral Norm
    str(res_dict['ComplexityType.PACBAYES_FLATNESS']) + ' ' +  # PAC-Bayesian Flatness
    str(res_dict['ComplexityType.LOG_PROD_OF_SPEC_OVER_MARGIN']) + ' ' +  # Spectral/Margin
    str(res_dict['ComplexityType.PACBAYES_INIT']) + ' ' +  # PB-I
    str(res_dict['ComplexityType.PACBAYES_ORIG']) + ' ' +  # PB-O
    str(res_dict['ComplexityType.PACBAYES_MAG_INIT']) + ' ' +  # PB-M-I
    str(res_dict['ComplexityType.PACBAYES_MAG_ORIG']) + ' ' +  # PB-M-O
    str(train_accuracy_list[-1]) + ' ' +
    str(test_accuracy_list[-1]) + ' ' +
    str(args.weight_decay) + ' ' +
    str(args.lr) + ' ' +
    str(args.batch_size) + '\n'
)

        if args.gradient == 1:
            variance = calculate_variance(net, trainloader, criterion, device)
            # Calculate and store the variance of gradients
            g = open(str(args.task) + '/' + str(args.net) + '_' + str(args.batch_size) + '.txt', 'a')
            g.write(str(average_region_list[-1]) + ' ' + str(variance.item()) + ' ' + str(args.weight_decay) + ' ' + str(args.lr) + '\n')
            r = open(str(args.task) + '/' + str(args.net) + '_' + str(args.lr) + '.txt', 'a')
            r.write(str(average_region_list[-1]) + ' ' + str(variance.item()) + ' ' + str(args.weight_decay) + ' ' + str(args.batch_size) + '\n')
        
        if args.scope_l != 0:
            f = open(str(args.net) + ' ' + str(args.scope_l) + '.txt', 'a')
            f.write(str(args.data_choose) + ' ' + ' '.join([str(elem) for elem in average_region_list]) + ' ' + str(train_accuracy_list[-1]) + ' ' + str(test_accuracy_list[-1]) + ' ' + str(args.weight_decay) + ' ' + str(args.lr) + ' ' + str(args.batch_size) + '\n')
        else:
            f = open(str(args.net) + '_' + args.dataset + '.txt', 'a')
            f.write(str(args.data_choose) + ' ' + ' '.join([str(elem) for elem in average_region_list]) + ' ' + str(train_accuracy_list[-1]) + ' ' + str(test_accuracy_list[-1]) + ' ' + str(args.weight_decay) + ' ' + str(args.lr) + ' ' + str(args.batch_size) + '\n')
        