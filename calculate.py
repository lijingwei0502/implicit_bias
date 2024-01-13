import torch
import numpy as np
import matplotlib.pyplot as plt
from projection import cal_dist, cal_proj, generalize_cal_dist, generalize_cal_proj


def test_gaussian(dim=100, data_size=15000, seed=0, var=1):
    std_deviation = torch.sqrt(torch.tensor(var))
    gaussian_dist = torch.distributions.normal.Normal(0.0, std_deviation)
    v1 = gaussian_dist.sample((dim,))
    v2 = gaussian_dist.sample((dim,))
    v3 = gaussian_dist.sample((dim,))

    v_arr = gaussian_dist.sample((data_size, dim))

    dist_arr = []
    
    for i, v in enumerate(v_arr):
        dist = cal_dist(v1, v2, v3, v)
        dist_arr.append((dist, i))

    sorted_dist = sorted(dist_arr, key=lambda x: x[0])

    out_num = 0
    tot_num = 0
    ratio_arr = []

    for dist, i in sorted_dist:
        _, x_coord, y_coord = cal_proj(v1, v2, v3, v_arr[i])

        # print(i, label, x_coord, y_coord)
        if x_coord < 0 or y_coord < 0 or x_coord + y_coord > 1:
            out_num += 1
            # print('out of triangle')
        tot_num += 1
        if tot_num % 100 == 0:
            print(tot_num, out_num, out_num / tot_num)
            ratio_arr.append(out_num / tot_num)

    x_arr = np.linspace(0, data_size, 100)
    plt.plot(x_arr, ratio_arr)
    plt.savefig('ratio_gaussian_var_' + str(var) + '_seed' + str(seed) + '.png')
    plt.close()


def test_cauchy(dim=100, data_size=15000, seed=0):
    location = 0.0
    scale = 1.0

    cauchy_dist = torch.distributions.cauchy.Cauchy(location, scale)

    v1 = cauchy_dist.sample((dim,))
    v2 = cauchy_dist.sample((dim,))
    v3 = cauchy_dist.sample((dim,))

    v_arr = cauchy_dist.sample((data_size, dim))

    dist_arr = []

    for i, v in enumerate(v_arr):
        dist = cal_dist(v1, v2, v3, v)
        dist_arr.append((dist, i))

    sorted_dist = sorted(dist_arr, key=lambda x: x[0])

    out_num = 0
    tot_num = 0
    ratio_arr = []

    for dist, i in sorted_dist:
        _, x_coord, y_coord = cal_proj(v1, v2, v3, v_arr[i])

        # print(i, label, x_coord, y_coord)
        if x_coord < 0 or y_coord < 0 or x_coord + y_coord > 1:
            out_num += 1
            # print('out of triangle')
        tot_num += 1
        if tot_num % 100 == 0:
            print(tot_num, out_num, out_num / tot_num)
            ratio_arr.append(out_num / tot_num)

    x_arr = np.linspace(0, data_size, data_size//100)
    plt.plot(x_arr, ratio_arr)
    plt.savefig('ratio_cauchy_' + str(seed) + '.png')
    plt.close()

def triangle(trainset_no_random, distance, samples):
    sample_1, sample_2, sample_3 = samples
    sorted_distances = sorted(distance, key=lambda x: x[0])

    out_num = 0
    tot_num = 0
    ratio_arr = []

    for distance, i in sorted_distances:
        sample, label = trainset_no_random[i]
        # 计算投影点

        projection_point, x_coord, y_coord = cal_proj(sample_1, sample_2, sample_3, sample)

        if x_coord < 0 or y_coord < 0 or x_coord + y_coord > 1:
            out_num += 1
        
        tot_num += 1
      
        if tot_num % 100 == 0:
            #print(tot_num, out_num, out_num / tot_num)
            ratio_arr.append(out_num / tot_num)

    x_arr = np.linspace(0, 15000, 150)
    plt.xlabel('Samples with distance from near to far')
    plt.ylabel('Ratio in triangle')
    plt.plot(x_arr, ratio_arr)
    plt.savefig('ratio.png')
    
def general_test_gaussian(nums):
    dim=3072
    data_size=1500
    var=1
    std_deviation = torch.sqrt(torch.tensor(var))
    gaussian_dist = torch.distributions.normal.Normal(0.0, std_deviation)
    v0 = []
    for i in range(nums):
        v = gaussian_dist.sample((dim,))
        v0.append(v)

    v_arr = gaussian_dist.sample((data_size, dim))

    dist_arr = []

    for i, v in enumerate(v_arr):
        dist = generalize_cal_dist(v0, v)
        dist_arr.append((dist, i))

    sorted_dist = sorted(dist_arr, key=lambda x: x[0])

    out_num = 0
    tot_num = 0
    ratio_arr = []

    for dist, i in sorted_dist:
        _, coords = generalize_cal_proj(v0, v_arr[i])

        # print(i, label, x_coord, y_coord)
        cnt = 0
        for coord in coords:
            if coord < 0:
                cnt += 1
                break
        if cnt or sum(coords) > 1:
            out_num += 1
            # print('out of triangle')
        tot_num += 1
        if tot_num % 100 == 0:
            ratio_arr.append(out_num / tot_num)

    x_arr = np.linspace(0, data_size, data_size//100)
    plt.plot(x_arr, ratio_arr)
    plt.savefig('gaussian_' + str(nums) + '.png')


def general_test_cauchy(nums):
    dim=3072
    data_size=15000
    location = 0.0
    scale = 1.0

    cauchy_dist = torch.distributions.cauchy.Cauchy(location, scale)

    v0 = []
    for i in range(nums):
        v = cauchy_dist.sample((dim,))
        v0.append(v)

    v_arr = cauchy_dist.sample((data_size, dim))

    dist_arr = []

    for i, v in enumerate(v_arr):
        dist = generalize_cal_dist(v0, v)
        dist_arr.append((dist, i))

    sorted_dist = sorted(dist_arr, key=lambda x: x[0])

    out_num = 0
    tot_num = 0
    ratio_arr = []

    for dist, i in sorted_dist:
        _, coords = generalize_cal_proj(v0, v_arr[i])

        # print(i, label, x_coord, y_coord)
        cnt = 0
        for coord in coords:
            if coord < 0:
                cnt += 1
                break
        if cnt or sum(coords) > 1:
            out_num += 1
            # print('out of triangle')
        tot_num += 1
        if tot_num % 100 == 0:
            ratio_arr.append(out_num / tot_num)

    x_arr = np.linspace(0, data_size, data_size//100)
    plt.plot(x_arr, ratio_arr)
    plt.savefig('cauchy' + str(nums) + '.png')
    
def general_triangle(trainset_no_random, samples):
    distance_list = []
    # 假设trainset是一个生成器或者列表，它迭代地给出(sample, label)对
    for i, (sample, label) in enumerate(trainset_no_random):
        # 计算点到平面的距离
        distance = generalize_cal_dist(samples, sample)
        distance_list.append((distance, i))
    
    sorted_distances = sorted(distance_list, key=lambda x: x[0])

    out_num = 0
    tot_num = 0
    ratio_arr = []

    for distance, i in sorted_distances:
        sample, label = trainset_no_random[i]
        # 计算投影点

        projection_point, coords = generalize_cal_proj(samples, sample)

        # 统计coords中的负数
        cnt = 0
        for coord in coords:
            if coord < 0:
                cnt += 1
                break
        if cnt or sum(coords) > 1:
            out_num += 1
        tot_num += 1
      
        if tot_num % 100 == 0:
            print(tot_num, out_num, out_num / tot_num)
            ratio_arr.append(out_num / tot_num)

    x_arr = np.linspace(0, 15000, 150)
    plt.xlabel('Samples with distance from near to far')
    plt.ylabel('Ratio in triangle')
    plt.plot(x_arr, ratio_arr)
    plt.savefig(str(len(samples)) + 'ratio.png')

def coincide(net, countnumbers, skipnumber, trainset_no_random, sample_1, sample_2, sample_3, device):
    plt.figure()
    distance_list = []
    # 假设trainset是一个生成器或者列表，它迭代地给出(sample, label)对
    for i, (sample, label) in enumerate(trainset_no_random):
        # 计算点到平面的距离
        distance = cal_dist(sample_1, sample_2, sample_3, sample)

        # 添加到距离列表
        distance_list.append((distance, i))
    sorted_distances = sorted(distance_list, key=lambda x: x[0])[:countnumbers]

    generated_samples = np.zeros((countnumbers, 3, 32, 32))
    net.eval()  
    labels = []
    cnt = 0
    for distance, i in sorted_distances:
        sample, label = trainset_no_random[i]
        # 计算投影点
        labels.append(label)
        projection_point, x_coord, y_coord = cal_proj(sample_1, sample_2, sample_3, sample)
        generated_sample = (1 - x_coord - y_coord) * sample_1 + x_coord * sample_2 + y_coord * sample_3
        generated_samples[cnt] = generated_sample
        cnt += 1
       
    input_data = torch.tensor(generated_samples, dtype=torch.float32).to(device)
    output = net(input_data)
    _, predictions = torch.max(output, 1)
    predictions = predictions.cpu().numpy()
    out_num = 0
    tot_num = 0
    ratio_arr = []
    for i in range(countnumbers):
        if predictions[i] == labels[i]:
            out_num += 1
        tot_num += 1
        #print(tot_num, out_num, out_num / tot_num)
        if i % skipnumber == 0:
            ratio_arr.append(out_num / tot_num)
    
    x_arr = np.linspace(0, countnumbers, countnumbers//skipnumber)
    plt.xlabel('Samples with distance from near to far')
    plt.ylabel('Ratio of the coincide')
    plt.plot(x_arr, ratio_arr)
    plt.savefig('ratio.png')
    
def n_coincide(net, countnumbers, trainset_no_random, samples, label_all, device, args):
    plt.figure()
    distance_list = []
    # 假设trainset是一个生成器或者列表，它迭代地给出(sample, label)对
    for i, (sample, label) in enumerate(trainset_no_random):
        # 计算点到平面的距离
        distance = generalize_cal_dist(samples, sample)

        # 添加到距离列表
        distance_list.append((distance, i))
    sorted_distances = sorted(distance_list, key=lambda x: x[0])[len(samples):countnumbers + len(samples)]

    generated_samples = np.zeros((countnumbers, 3, 32, 32))
    net.eval()  
    labels = []
    cnt = 0
    cnt1 = 0
    cnt2 = 0
    cnt3 = 0
    for distance, i in sorted_distances:
        sample, label = trainset_no_random[i]
        # 计算投影点
        labels.append(label)
        if label == 0:
            cnt1 += 1
        elif label == 1:
            cnt2 += 1
        else:
            cnt3 += 1
        projection_point, coords = generalize_cal_proj(samples, sample)
        generated_sample = (1 - sum(coords)) * samples[0]
        for i in range(1, len(coords)):
            generated_sample += coords[i] * samples[i]
        generated_samples[cnt] = generated_sample
        cnt += 1
       
    input_data = torch.tensor(generated_samples, dtype=torch.float32).to(device)
    output = net(input_data)
    _, predictions = torch.max(output, 1)
    predictions = predictions.cpu().numpy()
    out_num = 0
    tot_num = 0
    ratio_arr = []
    for i in range(countnumbers):
        if predictions[i] == labels[i]:
            out_num += 1
        tot_num += 1
        ratio_arr.append(out_num / tot_num)

    # change label_all to a string
    label_all = [str(x) for x in label_all]
    label_all = ','.join(label_all)
    # x_arr = np.linspace(0, countnumbers, countnumbers)
    # plt.xlabel('Samples with distance from near to far')
    # plt.ylabel('Ratio of the coincide')
    # plt.plot(x_arr, ratio_arr)
    save_dir = args.dir + "/results.csv" 
    # with open(save_dir, 'a') as f:
    #     f.write(label_all+','+str(cnt1)+','+str(cnt2)+','+str(cnt3)+','+str(args.seed) + "," + str(args.generate_point_num)+"," + str(args.count_numbers)+ ","+str(out_num/tot_num)+"\n")
    with open(save_dir, 'a') as f:
        f.write(str(args.seed) + "," + str(args.generate_point_num)+"," + str(args.count_numbers)+ ","+str(out_num/tot_num)+"\n")


def seperate_coincide(net, countnumbers, skipnumber, trainset_no_random, sample_1, sample_2, sample_3, device):
    plt.figure()
    distance_list = {}
    # 假设trainset是一个生成器或者列表，它迭代地给出(sample, label)对
    for i, (sample, label) in enumerate(trainset_no_random):
        # 计算点到平面的距离
        distance = cal_dist(sample_1, sample_2, sample_3, sample)
        # 添加到距离列表
        if label not in distance_list:
            distance_list[label] = []
        distance_list[label].append((distance, i))
    
    for label in distance_list.keys():
        sorted_distances = sorted(distance_list[label], key=lambda x: x[0])[:countnumbers]
        generated_samples = np.zeros((countnumbers, 3, 32, 32))
        net.eval()  
        labels = []
        cnt = 0
        for distance, i in sorted_distances:
            sample, label_now = trainset_no_random[i]
            # 计算投影点
            labels.append(label_now)
            projection_point, x_coord, y_coord = cal_proj(sample_1, sample_2, sample_3, sample)
            #print(i, label, x_coord, y_coord)
            generated_sample = (1 - x_coord - y_coord) * sample_1 + x_coord * sample_2 + y_coord * sample_3
            generated_samples[cnt] = generated_sample
            cnt += 1
        
        input_data = torch.tensor(generated_samples, dtype=torch.float32).to(device)
        output = net(input_data)
        _, predictions = torch.max(output, 1)
        predictions = predictions.cpu().numpy()
        out_num = 0
        tot_num = 0
        ratio_arr = []
        for i in range(countnumbers):
            #print(predictions[i], labels[i])
            if predictions[i] == labels[i]:
                out_num += 1
            tot_num += 1
            #print(tot_num, out_num, out_num / tot_num)
            if i % skipnumber == 0:
                ratio_arr.append(out_num / tot_num)
        
        x_arr = np.linspace(0, countnumbers, countnumbers//skipnumber)
        plt.xlabel('Samples with distance from near to far')
        plt.ylabel('Ratio of the coincide')
        plt.plot(x_arr, ratio_arr, label='class ' + str(label))
    plt.legend()
    plt.savefig('seperate_ratio.png')  
    
    plt.figure()
    for label in distance_list.keys():
        sorted_distances = sorted(distance_list[label], key=lambda x: x[0])[:countnumbers]
        dist_arr = []
        cnt = 0
        for distance, i in sorted_distances:
            cnt += 1
            if cnt % skipnumber == 0:
                dist_arr.append(distance)
        x_arr = np.linspace(0, countnumbers, countnumbers//skipnumber)
        plt.xlabel('Samples with distance from near to far')
        plt.ylabel('distance')
        plt.plot(x_arr, dist_arr, label = 'class ' + str(label))
    plt.legend()
    plt.savefig('seperate_distance.png')  
    
def epochcoincide(net, countnumbers, ratio_arr, trainset_no_random, sample_1, sample_2, sample_3, device):
    distance_list = {}
    # 假设trainset是一个生成器或者列表，它迭代地给出(sample, label)对
    for i, (sample, label) in enumerate(trainset_no_random):
        # 计算点到平面的距离
        distance = cal_dist(sample_1, sample_2, sample_3, sample)
        # 添加到距离列表
        if label not in distance_list:
            distance_list[label] = []
        distance_list[label].append((distance, i))
    
    for label in distance_list.keys():
        sorted_distances = sorted(distance_list[label], key=lambda x: x[0])[:countnumbers]
        generated_samples = np.zeros((countnumbers, 3, 32, 32))
        net.eval()  
        labels = []
        cnt = 0
        for distance, i in sorted_distances:
            sample, label_now = trainset_no_random[i]
            # 计算投影点
            labels.append(label_now)
            projection_point, x_coord, y_coord = cal_proj(sample_1, sample_2, sample_3, sample)
            #print(i, label, x_coord, y_coord)
            generated_sample = (1 - x_coord - y_coord) * sample_1 + x_coord * sample_2 + y_coord * sample_3
            generated_samples[cnt] = generated_sample
            cnt += 1
        
        input_data = torch.tensor(generated_samples, dtype=torch.float32).to(device)
        output = net(input_data)
        _, predictions = torch.max(output, 1)
        predictions = predictions.cpu().numpy()
        out_num = 0
        tot_num = 0
        
        for i in range(countnumbers):
            #print(predictions[i], labels[i])
            if predictions[i] == labels[i]:
                out_num += 1
            tot_num += 1
            
        if label not in ratio_arr:
            ratio_arr[label] = []
        ratio_arr[label].append(out_num/tot_num)
        
    return ratio_arr

if __name__ == 'main':
    test_gaussian(seed=0)
    test_gaussian(seed=1)
    test_gaussian(seed=2)
    test_gaussian(seed=3)
    test_gaussian(seed=4)
    test_cauchy(seed=0)
    test_cauchy(seed=1)
    test_cauchy(seed=2)
    test_cauchy(seed=3)
    test_cauchy(seed=4)

    test_gaussian(var=1)
    test_gaussian(var=2)
    test_gaussian(var=5)
    test_gaussian(var=10)
    test_gaussian(var=20)
    test_gaussian(var=50)
    test_gaussian(var=100)