import torch
import numpy as np

def cal_line(prediction_line):
    mark_line = np.zeros(prediction_line.shape, dtype = 'int64')
    mark_num = 0
    w = prediction_line.shape[0]
    direct_delta = [-1, 1]
    all_kinds = 0
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
    return all_kinds

def cal_2d_matrix(prediction_matrix):
    mark_matrix = np.zeros(prediction_matrix.shape, dtype = 'int64')
    mark_num = 0
    w, h = prediction_matrix.shape[0], prediction_matrix.shape[1]
    direct_delta = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]
    all_kinds = 0
    for i in range(w):
        for j in range(h):
            if mark_matrix[i][j] > 0 or prediction_matrix[i][j] == -1:
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
    return all_kinds

def cal_3d_matrix(prediction_matrix):
    mark_matrix = np.zeros(prediction_matrix.shape, dtype = 'int64')
    mark_num = 0
    w, h, d = prediction_matrix.shape[0], prediction_matrix.shape[1], prediction_matrix.shape[2]
    direct_delta = [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]]
    all_kinds = 0
    for i in range(w):
        for j in range(h):
            for k in range(d):
                if mark_matrix[i][j][k] > 0 or prediction_matrix[i][j][k] == -1:
                    continue
                queue = [[i, j, k]]
                mark_num += 1
                mark_matrix[i][j][k] = mark_num
                tnt = 0
                while len(queue) > 0:
                    tnt += 1
                    cur_x, cur_y, cur_z = queue[0]
                    queue.pop(0)
                    for delta_x, delta_y, delta_z in direct_delta:
                        tmp_x = cur_x + delta_x
                        tmp_y = cur_y + delta_y
                        tmp_z = cur_z + delta_z
                        if tmp_x < 0 or tmp_x >= w or tmp_y < 0 or tmp_y >= h or tmp_z < 0 or tmp_z >= d or mark_matrix[tmp_x][tmp_y][tmp_z] > 0:
                            continue
                        if prediction_matrix[tmp_x][tmp_y][tmp_z] == prediction_matrix[cur_x][cur_y][cur_z]:
                            mark_matrix[tmp_x][tmp_y][tmp_z] = mark_matrix[cur_x][cur_y][cur_z]
                            queue.append([tmp_x, tmp_y, tmp_z])
                all_kinds += 1
    return all_kinds

def cal_4d_matrix(prediction_matrix):
    mark_matrix = np.zeros(prediction_matrix.shape, dtype = 'int64')
    mark_num = 0
    w, h, d, k = prediction_matrix.shape[0], prediction_matrix.shape[1], prediction_matrix.shape[2], prediction_matrix.shape[3]
    direct_delta = [[-1, 0, 0, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 1, 0], [0, 0, 0, -1], [0, 0, 0, 1]]
    all_kinds = 0
    for i in range(w):
        for j in range(h):
            for l in range(d):
                for m in range(k):
                    if mark_matrix[i][j][l][m] > 0 or prediction_matrix[i][j][l][m] == -1:
                        continue
                    queue = [[i, j, l, m]]
                    mark_num += 1
                    mark_matrix[i][j][l][m] = mark_num
                    tnt = 0
                    while len(queue) > 0:
                        tnt += 1
                        cur_x, cur_y, cur_z, cur_k = queue[0]
                        queue.pop(0)
                        for delta_x, delta_y, delta_z, delta_k in direct_delta:
                            tmp_x = cur_x + delta_x
                            tmp_y = cur_y + delta_y
                            tmp_z = cur_z + delta_z
                            tmp_k = cur_k + delta_k
                            if tmp_x < 0 or tmp_x >= w or tmp_y < 0 or tmp_y >= h or tmp_z < 0 or tmp_z >= d or tmp_k < 0 or tmp_k >= k or mark_matrix[tmp_x][tmp_y][tmp_z][tmp_k] > 0:
                                continue
                            if prediction_matrix[tmp_x][tmp_y][tmp_z][tmp_k] == prediction_matrix[cur_x][cur_y][cur_z][cur_k]:
                                mark_matrix[tmp_x][tmp_y][tmp_z][tmp_k] = mark_matrix[cur_x][cur_y][cur_z][cur_k]
                                queue.append([tmp_x, tmp_y, tmp_z, tmp_k])
                    all_kinds += 1
    return all_kinds

def cal_5d_matrix(prediction_matrix):
    mark_matrix = np.zeros(prediction_matrix.shape, dtype = 'int64')
    mark_num = 0
    w, h, d, k, l = prediction_matrix.shape[0], prediction_matrix.shape[1], prediction_matrix.shape[2], prediction_matrix.shape[3], prediction_matrix.shape[4]
    direct_delta = [[-1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, -1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, -1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, -1, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, -1], [0, 0, 0, 0, 1]]
    all_kinds = 0
    for i in range(w):
        for j in range(h):
            for m in range(d):
                for n in range(k):
                    for o in range(l):
                        if mark_matrix[i][j][m][n][o] > 0 or prediction_matrix[i][j][m][n][o] == -1:
                            continue
                        queue = [[i, j, m, n, o]]
                        mark_num += 1
                        mark_matrix[i][j][m][n][o] = mark_num
                        tnt = 0
                        while len(queue) > 0:
                            tnt += 1
                            cur_x, cur_y, cur_z, cur_k, cur_l = queue[0]
                            queue.pop(0)
                            for delta_x, delta_y, delta_z, delta_k, delta_l in direct_delta:
                                tmp_x = cur_x + delta_x
                                tmp_y = cur_y + delta_y
                                tmp_z = cur_z + delta_z
                                tmp_k = cur_k + delta_k
                                tmp_l = cur_l + delta_l
                                if tmp_x < 0 or tmp_x >= w or tmp_y < 0 or tmp_y >= h or tmp_z < 0 or tmp_z >= d or tmp_k < 0 or tmp_k >=d or tmp_l < 0 or tmp_l >= l or mark_matrix[tmp_x][tmp_y][tmp_z][tmp_k][tmp_l] > 0:
                                    continue
                                if prediction_matrix[tmp_x][tmp_y][tmp_z][tmp_k][tmp_l] == prediction_matrix[cur_x][cur_y][cur_z][cur_k][cur_l]:
                                    mark_matrix[tmp_x][tmp_y][tmp_z][tmp_k][tmp_l] = mark_matrix[cur_x][cur_y][cur_z][cur_k][cur_l]
                                    queue.append([tmp_x, tmp_y, tmp_z, tmp_k, tmp_l])
                        all_kinds += 1

    return all_kinds

def calculate_region(args, regions_list, device, model, samples_list):
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
                regions = cal_line(predictions)
                regions_list.append(regions)

        elif args.data_choose == 3:
            x_min, x_max = args.scope_l, args.scope_r
            y_min, y_max = args.scope_l, args.scope_r
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, num=10),
                                np.linspace(y_min, y_max, num=10))
            num_points = xx.ravel().shape[0]
            cnt = 0

            for sample_1, sample_2, sample_3 in samples_list:
                cnt += 1
                generated_samples = []
                mask = []
                for i in range(num_points):
                    alpha, beta = xx.ravel()[i], yy.ravel()[i]
                    if alpha + beta <= 1:
                        generated_sample = (1 - alpha - beta) * sample_1 + alpha * sample_2 + beta * sample_3
                        generated_samples.append(generated_sample)
                        mask.append(True)
                    else:
                        if args.plot:
                            generated_sample = (1 - alpha - beta) * sample_1 + alpha * sample_2 + beta * sample_3
                            generated_samples.append(generated_sample)
                            mask.append(True)
                        else:
                            mask.append(False)
                generated_samples = np.array(generated_samples)

                input_data = torch.tensor(generated_samples, dtype=torch.float32).to(device)
                output = model(input_data)
                _, predictions = torch.max(output, 1)
                predictions = predictions.cpu().numpy()

                mask = np.array(mask).reshape(xx.shape)
                predictions_full = np.full(xx.shape, -1)  
                predictions_full[mask] = predictions

                regions = cal_2d_matrix(predictions_full)
                regions_list.append(regions)

        elif args.data_choose == 4:
            x_min, x_max = args.scope_l, args.scope_r
            y_min, y_max = args.scope_l, args.scope_r
            z_min, z_max = args.scope_l, args.scope_r
            xx, yy, zz = np.meshgrid(np.linspace(x_min, x_max, num=5),
                                    np.linspace(y_min, y_max, num=5),
                                    np.linspace(z_min, z_max, num=5))
            num_points = xx.ravel().shape[0]
            cnt = 0
            for sample_1, sample_2, sample_3, sample_4 in samples_list:
                cnt += 1
                mask = []
                generated_samples = []
                for i in range(num_points):
                    alpha, beta, gamma = xx.ravel()[i], yy.ravel()[i], zz.ravel()[i]
                    if alpha + beta + gamma <= 1:
                        generated_sample = (1 - alpha - beta - gamma) * sample_1 + alpha * sample_2 + beta * sample_3 + gamma * sample_4
                        generated_samples.append(generated_sample)
                        mask.append(True)
                    else:
                        mask.append(False)
                generated_samples = np.array(generated_samples)
                input_data = torch.tensor(generated_samples, dtype=torch.float32).to(device)
                output = model(input_data)
                _, predictions = torch.max(output, 1)
                predictions = predictions.cpu().numpy()
                mask = np.array(mask).reshape(xx.shape)
                predictions_full = np.full(xx.shape, -1)  
                predictions_full[mask] = predictions
                regions = cal_3d_matrix(predictions_full)
                regions_list.append(regions)
            
        elif args.data_choose == 5:
            x_min, x_max = args.scope_l, args.scope_r
            y_min, y_max = args.scope_l, args.scope_r
            z_min, z_max = args.scope_l, args.scope_r
            w_min, w_max = args.scope_l, args.scope_r
            xx, yy, zz, ww = np.meshgrid(np.linspace(x_min, x_max, num=5),
                                    np.linspace(y_min, y_max, num=5),
                                    np.linspace(z_min, z_max, num=5),
                                    np.linspace(w_min, w_max, num=5))
            num_points = xx.ravel().shape[0]
            cnt = 0
            for sample_1, sample_2, sample_3, sample_4, sample_5 in samples_list:
                cnt += 1
                generated_samples = []
                mask = []
                for i in range(num_points):
                    alpha, beta, gamma, delta = xx.ravel()[i], yy.ravel()[i], zz.ravel()[i], ww.ravel()[i]
                    if alpha + beta + gamma + delta <= 1:
                        generated_sample = (1 - alpha - beta - gamma - delta) * sample_1 + alpha * sample_2 + beta * sample_3 + gamma * sample_4 + delta * sample_5
                        generated_samples.append(generated_sample)
                        mask.append(True)
                    else:
                        mask.append(False)
                generated_samples = np.array(generated_samples)
                input_data = torch.tensor(generated_samples, dtype=torch.float32).to(device)
                output = model(input_data)
                _, predictions = torch.max(output, 1)
                predictions = predictions.cpu().numpy()
                mask = np.array(mask).reshape(xx.shape)
                predictions_full = np.full(xx.shape, -1)
                predictions_full[mask] = predictions
                regions = cal_4d_matrix(predictions_full)
                regions_list.append(regions)

        elif args.data_choose == 6:
            x_min, x_max = args.scope_l, args.scope_r
            y_min, y_max = args.scope_l, args.scope_r
            z_min, z_max = args.scope_l, args.scope_r
            w_min, w_max = args.scope_l, args.scope_r
            k_min, k_max = args.scope_l, args.scope_r
            xx, yy, zz, ww, kk = np.meshgrid(np.linspace(x_min, x_max, num=5),
                                    np.linspace(y_min, y_max, num=5),
                                    np.linspace(z_min, z_max, num=5),
                                    np.linspace(w_min, w_max, num=5),
                                    np.linspace(k_min, k_max, num=5))
            num_points = xx.ravel().shape[0]
            cnt = 0
            for sample_1, sample_2, sample_3, sample_4, sample_5, sample_6 in samples_list:
                cnt += 1
                generated_samples = []
                mask = []
                for i in range(num_points):
                    alpha, beta, gamma, delta, epsilon = xx.ravel()[i], yy.ravel()[i], zz.ravel()[i], ww.ravel()[i], kk.ravel()[i]
                    if alpha + beta + gamma + delta + epsilon <= 1:
                        generated_sample = (1 - alpha - beta - gamma - delta - epsilon) * sample_1 + alpha * sample_2 + beta * sample_3 + gamma * sample_4 + delta * sample_5 + epsilon * sample_6
                        generated_samples.append(generated_sample)
                        mask.append(True)
                    else:
                        mask.append(False)
                generated_samples = np.array(generated_samples)
                input_data = torch.tensor(generated_samples, dtype=torch.float32).to(device)
                output = model(input_data)
                _, predictions = torch.max(output, 1)
                predictions = predictions.cpu().numpy()
                mask = np.array(mask).reshape(xx.shape)
                predictions_full = np.full(xx.shape, -1)
                predictions_full[mask] = predictions
                regions = cal_5d_matrix(predictions_full)
                regions_list.append(regions)


