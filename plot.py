import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

def calculate_region(args, epoch, regions_list, entropy_list, device, model, samples_list):
    model.eval()
    with torch.no_grad():
        if args.data_choose <=2:
            x_min, x_max = args.scope_l, args.scope_r
            xx = np.linspace(x_min, x_max, num=200)
            num_points = len(xx)
            cnt = 0
            for sample_1, sample_2 in samples_list:
                cnt += 1
                generated_samples = np.zeros((num_points, 3, 32, 32))
                for i in range(num_points):
                    alpha = xx[i]
                    generated_sample = (1 - alpha) * sample_1 + alpha * sample_2
                    generated_samples[i] = generated_sample
                input_data = torch.tensor(generated_samples, dtype=torch.float32).to(device)
                output = model(input_data)
                _, predictions = torch.max(output, 1)
                predictions = predictions.cpu().numpy()
                predictions = predictions.reshape(xx.shape)
                regions, entropy = cal_line(predictions)
                regions_list.append(regions)
                entropy_list.append(entropy)
        else:
            x_min, x_max = args.scope_l, args.scope_r
            y_min, y_max = args.scope_l, args.scope_r
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, num=30),
                                np.linspace(y_min, y_max, num=30))
            num_points = xx.ravel().shape[0]
            cnt = 0
            for sample_1, sample_2, sample_3 in samples_list:
                cnt += 1
                generated_samples = np.zeros((num_points, 3, 32, 32))
                for i in range(num_points):
                    alpha, beta = xx.ravel()[i], yy.ravel()[i]
                    generated_sample = (1 - alpha - beta) * sample_1 + alpha * sample_2 + beta * sample_3
                    generated_samples[i] = generated_sample
                input_data = torch.tensor(generated_samples, dtype=torch.float32).to(device)
                output = model(input_data)
                _, predictions = torch.max(output, 1)
                predictions = predictions.cpu().numpy()
                predictions = predictions.reshape(xx.shape)
                regions, entropy = cal_componet_entropy(predictions)
                regions_list.append(regions)
                entropy_list.append(entropy)
                if args.plot and cnt %20 == 0:
                    plt.figure(figsize=(5, 5))
                    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'coral', 'white', 'orange', 'purple']
                    num_classes = 10
                    class_colors = colors[:num_classes]
                    cmap = ListedColormap(class_colors)
                    norm = BoundaryNorm(boundaries=np.arange(num_classes + 1), ncolors=num_classes)
                    plt.contourf(xx, yy, predictions, cmap=cmap, norm=norm, levels=np.arange(num_classes+1)-0.5)
                    plt.tick_params(axis='both', which='both', length=0, fontsize=14)
                    plt.xlabel(r'$\alpha$', fontsize=18, labelpad=3)  # Alpha for the x-axis
                    plt.ylabel(r'$\beta$', fontsize=18, labelpad=3)   # Beta for the y-axis
                    # 按照epoch数和cnt//20保存图片
                    plt.savefig(args.dir + f'/epoch_{epoch}_cnt_{cnt//20}.png')
                            
def cal_componet_entropy(prediction_matrix):
    mark_matrix = np.zeros(prediction_matrix.shape, dtype = 'int64')
    mark_num = 0
    w, h = prediction_matrix.shape[0], prediction_matrix.shape[1]
    direct_delta = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]
    all_kinds = 0
    space = w*h
    entropy = 0
    for i in range(w):
        for j in range(h):
            if mark_matrix[i][j] > 0:
                continue
            queue = [[i, j]]
            mark_num += 1
            mark_matrix[i][j] = mark_num
            tnt = 0
            while len(queue) > 0:
                tnt += 1
                cur_x, cur_y = queue[0]
                queue.pop(0)
                for delta_x, delta_y in direct_delta:
                    tmp_x = cur_x + delta_x
                    tmp_y = cur_y + delta_y
                    if tmp_x < 0 or tmp_x >= w or tmp_y < 0 or tmp_y >= h or mark_matrix[tmp_x][tmp_y] > 0:
                        continue
                    if prediction_matrix[tmp_x][tmp_y] == prediction_matrix[cur_x][cur_y]:
                        mark_matrix[tmp_x][tmp_y] = mark_matrix[cur_x][cur_y]
                        queue.append([tmp_x, tmp_y])
            all_kinds += 1
            entropy += tnt/space * np.log(tnt/space)
    return all_kinds, -entropy

def cal_line(prediction_line):
    mark_line = np.zeros(prediction_line.shape, dtype = 'int64')
    mark_num = 0
    w = prediction_line.shape[0]
    direct_delta = [-1, 1]
    all_kinds = 0
    space = w
    entropy = 0
    for i in range(w):
        if mark_line[i] > 0:
            continue
        queue = [i]
        mark_num += 1
        mark_line[i] = mark_num
        tnt = 0
        while len(queue) > 0:
            tnt += 1
            cur_x = queue[0]
            queue.pop(0)
            for delta_x in direct_delta:
                tmp_x = cur_x + delta_x
                if tmp_x < 0 or tmp_x >= w or mark_line[tmp_x] > 0:
                    continue
                if prediction_line[tmp_x] == prediction_line[cur_x]:
                    mark_line[tmp_x] = mark_line[cur_x]
                    queue.append(tmp_x)
        all_kinds += 1
        entropy += tnt/space * np.log(tnt/space)
    return all_kinds, -entropy

def plot_loss_accuracy(args, start_epoch, num_epochs, average_region_list, average_entropy_list, variance_region_list, variance_entropy_list, train_loss_list, test_loss_list, train_accuracy_list, test_accuracy_list):
    
    plt.figure()
    plt.plot(range(start_epoch, start_epoch + num_epochs + 1, args.skip_plot), average_region_list, label='Average Regions')
    # plt.fill_between(range(start_epoch, start_epoch + num_epochs + 1, args.skip_plot), 
    #                 np.array(average_region_list) - np.array(variance_region_list), 
    #                 np.array(average_region_list) + np.array(variance_region_list), 
    #                 alpha=0.5, label='Variance')
    plt.title('Average number of regions over Epochs', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel('Epochs', fontsize=18)
    plt.ylabel('Average number of regions', fontsize=18)
    plt.legend()  # Now this will work because elements have labels
    plt.savefig(args.dir + '/average_region.png')
    plt.close()

    # plt.figure()
    # plt.plot(range(start_epoch, start_epoch + num_epochs + 1, args.skip_plot), average_entropy_list, label='Average Entropy')
    # plt.fill_between(range(start_epoch, start_epoch + num_epochs + 1, args.skip_plot),
    #                 np.array(average_entropy_list) - np.array(variance_entropy_list),
    #                 np.array(average_entropy_list) + np.array(variance_entropy_list),
    #                 alpha=0.5, label='Variance')
    # plt.title('Average Entropy over Epochs')
    # plt.xlabel('Epochs')
    # plt.ylabel('Average Entropy')
    # plt.legend()  # Now this will work because elements have labels
    # plt.savefig(args.dir + '/average_entropy.png')
    # plt.close()

    plt.figure()
    plt.plot(range(start_epoch, start_epoch + num_epochs + 1), train_loss_list, label='Train Loss')
    plt.plot(range(start_epoch, start_epoch + num_epochs + 1), test_loss_list, label='Test Loss')
    plt.title('Train and Test Loss Over Epochs', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel('Epochs', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.savefig(args.dir + '/loss_curve.png') 
    plt.close()  

    plt.figure()
    plt.plot(range(start_epoch, start_epoch + num_epochs + 1), train_accuracy_list, label='Train Accuracy')
    plt.plot(range(start_epoch, start_epoch + num_epochs + 1), test_accuracy_list, label='Test Accuracy')
    plt.title('Train and Test Accuracy Over Epochs', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel('Epochs', fontsize=18)
    plt.ylabel('Accuracy (%)', fontsize=18)
    plt.savefig(args.dir + '/accuracy_curve.png') 
    plt.close() 


            