from projection import cal_dist, cal_proj, generalize_cal_dist, generalize_cal_proj
import numpy as np

def generate_sample(trainset_no_random):
# 获取属于不同类别的随机样本的索引
    samples = []
    # 随机取三个点
    random_indices = np.random.choice(len(trainset_no_random), 3, replace=False)
    for index in random_indices:
        sample, label = trainset_no_random[index]
        samples.append(sample)

    sample_1, sample_2, sample_3 = samples
    return sample_1, sample_2, sample_3


def generate_nearest_sample(args, nearest_sample_numbers, trainset_no_random, sample_1, sample_2, sample_3):
    # 初始化一个空列表用于存储距离和索引
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
        distance_list[label] = sorted(distance_list[label], key=lambda x: x[0])[:nearest_sample_numbers]
    close_x = []
    close_y = []
    close_label = []
    for label in distance_list.keys():
        for distance, i in distance_list[label]:
            sample, label = trainset_no_random[i]
            projection_point, x_coord, y_coord = cal_proj(sample_1, sample_2, sample_3, sample)
            if distance < args.near_distance:
                close_x.append(x_coord)
                close_y.append(y_coord)
                close_label.append(label)

    return close_x, close_y, close_label

def calculate_decision_center(trainset_no_random, sample_1, sample_2, sample_3, average_numbers):
    # 初始化一个空列表用于存储距离和索引
    distance_list = []
    # 假设trainset是一个生成器或者列表，它迭代地给出(sample, label)对
    for i, (sample, label) in enumerate(trainset_no_random):
        # 计算点到平面的距离
        distance = cal_dist(sample_1, sample_2, sample_3, sample)
        distance_list.append((distance, i))
    
    # 对距离列表进行排序
    distance_list = sorted(distance_list, key=lambda x: x[0])[3:average_numbers]

    close_x = []
    close_y = []
    close_label = []
    for distance, i in distance_list:
        sample, label = trainset_no_random[i]
        projection_point, x_coord, y_coord = cal_proj(sample_1, sample_2, sample_3, sample)
        close_x.append(x_coord)
        close_y.append(y_coord)
        close_label.append(label)
       
    return close_x, close_y, close_label, distance_list
    
def n_generate_nearest_sample(trainset_no_random, samples, average_numbers):
    # 初始化一个空列表用于存储距离和索引
    distance_list = []
    # 假设trainset是一个生成器或者列表，它迭代地给出(sample, label)对
    for i, (sample, label) in enumerate(trainset_no_random):
        # 计算点到平面的距离
        #print(samples)
        distance = generalize_cal_dist(samples, sample)
        distance_list.append((distance, i))
    
    # 对距离列表进行排序
    distance_list = sorted(distance_list, key=lambda x: x[0])[:average_numbers]
    #print(distance_list)
    close_pos = []
    close_label = []
    for distance, i in distance_list:
        sample, label = trainset_no_random[i]
        projection_point, coord = generalize_cal_proj(samples, sample)
        close_pos.append(coord)
        close_label.append(label)
       
    # print(close_pos, close_label)
    return close_pos, close_label, distance_list