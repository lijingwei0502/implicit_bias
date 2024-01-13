import numpy as np



def cal_dist(v1, v2, v3, v): # 计算点v到平面的距离
    v1 = v1.flatten().numpy()
    v2 = v2.flatten().numpy()
    v3 = v3.flatten().numpy()
    v = v.flatten().numpy()
    e1 = (v2 - v1) / np.linalg.norm(v2 - v1)
    e2 = v3 - v1
    e2 = e2 - np.dot(e2, e1) * e1
    e2 = e2 / np.linalg.norm(e2)
    dist_vec = v - v1 - np.dot(v - v1, e1) * e1 - np.dot(v - v1, e2) * e2
    return np.linalg.norm(dist_vec)

def generalize_cal_dist(vector, v): # 广义计算距离
    # vectors是一个列表，每个元素是一个向量
    # v是一个向量
    vectors = vector.copy()
    for i in range(len(vectors)):
        vectors[i] = vectors[i].flatten().numpy()
    v= v.flatten().numpy()
    e = []
    for i in range(1, len(vectors)):
        ei = vectors[i] - vectors[0]
        for ej in e:
            ei = ei - np.dot(ei, ej) * ej
        ei = ei / np.linalg.norm(ei)
        e.append(ei)
    dist_vec = v - vectors[0]
    for ei in e:
        dist_vec = dist_vec - np.dot(dist_vec, ei) * ei
    return np.linalg.norm(dist_vec)

def cal_proj(v1, v2, v3, v): # 计算点v到平面的投影
    v1 = v1.flatten().numpy()
    v2 = v2.flatten().numpy()
    v3 = v3.flatten().numpy()
    v = v.flatten().numpy()
    e1 = (v2 - v1) / np.linalg.norm(v2 - v1)
    e2 = v3 - v1
    e2 = e2 - np.dot(e2, e1) * e1
    e2 = e2 / np.linalg.norm(e2)
    proj_point = v1 + np.dot(v - v1, e1) * e1 + np.dot(v - v1, e2) * e2
    tmp1 = np.dot(v - v1, e1)
    tmp2 = np.dot(v - v1, e2)
    tmp3 = np.dot(v3 - v1, e1)
    tmp4 = np.linalg.norm((v3 - v1) - np.dot(v3 - v1, e1) * e1)
    tmp5 = np.linalg.norm(v2 - v1)
    x_coord = (tmp1 - tmp2 * tmp3 / tmp4) / tmp5
    y_coord = tmp2 / tmp4
    return proj_point, x_coord, y_coord

def generalize_cal_proj(vector, v):
    vectors = vector.copy()
    for i in range(len(vectors)):
        vectors[i] = vectors[i].flatten().numpy()
    v= v.flatten().numpy()
    n = len(vectors)
    e = []
    for i in range(1, len(vectors)):
        ei = vectors[i] - vectors[0]
        for ej in e:
            ei = ei - np.dot(ei, ej) * ej
        ei = ei / np.linalg.norm(ei)
        e.append(ei)
    proj_point = vectors[0]
    for i in range(n-1):
        proj_point = proj_point + np.dot(v - vectors[0], e[i]) * e[i]
    x = np.zeros(n-1)
    # 从n-2到0遍历
    for i in range(n-2, -1, -1):
        x[i] = np.dot(v - vectors[0], e[i])
        for j in range(n-2, i, -1):
            x[i] = x[i] - x[j]*np.dot(vectors[j+1] - vectors[0], e[i])
        x[i] = x[i] / np.dot(vectors[i+1] - vectors[0], e[i])
    
    return proj_point, x
    

def find_random_points(trainset_no_random, v, pointsnum):
    v= v.flatten().numpy().copy()
    dist = []
    for i in range(0, len(trainset_no_random)):
        dist.append(np.linalg.norm(trainset_no_random[i][0].flatten().numpy() - v))
    dist = np.array(dist)
    # index 为随机抽取的10个点
    index = np.random.randint(0, len(trainset_no_random), pointsnum)
    #print(index)
    samples = []
    label = []
    for i in range(0,pointsnum):
        samples.append(trainset_no_random[index[i]][0])
        label.append(trainset_no_random[index[i]][1])
    return samples, label

def find_nearest_plane(trainset_no_random, v):
    
    # 在trainset_no_random中找到距离v最近的三个点

    dist = []
    for i in range(0, len(trainset_no_random)):
        dist.append(np.linalg.norm(trainset_no_random[i][0].flatten().numpy() - v.flatten().numpy()))
    
    dist = np.array(dist)
    index = np.argsort(dist)
    # 取出离得最近的100个点
    num = 50
    index = index[1:num]
    # 从这100个点中枚举所有的三个点
    min_dist = 100000000
    label_final = []
    min_sample_1 = trainset_no_random[index[0]][0]
    min_sample_2 = trainset_no_random[index[0]][0]
    min_sample_3 = trainset_no_random[index[0]][0]
    for i in range(0,   num-1):
        for j in range(i + 1,   num-1):
            for k in range(j + 1,  num-1):
                sample_1 = trainset_no_random[index[i]][0]
                sample_2 = trainset_no_random[index[j]][0]
                sample_3 = trainset_no_random[index[k]][0]
                dist = cal_dist(sample_1, sample_2, sample_3, v)
                if dist < min_dist:
                    min_dist = dist
                    min_sample_1 = sample_1
                    min_sample_2 = sample_2
                    min_sample_3 = sample_3
                    label = []
                    label.append(trainset_no_random[index[i]][1])
                    label.append(trainset_no_random[index[j]][1])
                    label.append(trainset_no_random[index[k]][1])
                    label_final = label

    return min_sample_1, min_sample_2, min_sample_3, label_final



def lasso_find_nearest_plane(trainset_no_random, v):
    
    # 在trainset_no_random中找到距离v最近的三个点
    y= v.flatten().numpy()
    #x_final = trainset_no_random[len(trainset_no_random)-1][0].flatten().numpy()
    #y -= x_final
    flattened_vectors = []

    # Iterate through trainset_no_random and flatten each vector
    for i in range(len(trainset_no_random)):  
        flattened_vector = trainset_no_random[i][0].flatten().numpy() 
        flattened_vectors.append(flattened_vector)

    # Convert the list of flattened vectors to a NumPy array
    X = np.array(flattened_vectors)
    X = X.T
    
    lasso = Lasso(alpha=0.1)
    lasso.fit(X, y)

    # 输出所有系数的和
    #print(f"Sum of coefficients: {np.sum(lasso.coef_)}")
    # 增加一个系数，为1-所有系数的和
    #lasso.coef_ = np.append(lasso.coef_, 1 - np.sum(lasso.coef_))
    
    #print(f"Number of non-zero coefficients: {np.sum(lasso.coef_ != 0)}")
    
    index = np.argsort(-lasso.coef_)
    # 把=0的index都删除
    index = index[lasso.coef_[index] != 0]
    #print(index)
    # 排序后重新输出所有非零系数
    #print(f"Coefficients: {lasso.coef_[index]}")
    
    num = len(index)
    # 从这100个点中枚举所有的三个点
    min_dist = 100000000
    label_final = []
    min_sample_1 = trainset_no_random[index[0]][0]
    min_sample_2 = trainset_no_random[index[0]][0]
    min_sample_3 = trainset_no_random[index[0]][0]
    for i in range(0,  num):
        for j in range(i + 1,   num):
            for k in range(j + 1,  num):
                sample_1 = trainset_no_random[index[i]][0]
                sample_2 = trainset_no_random[index[j]][0]
                sample_3 = trainset_no_random[index[k]][0]
                dist = cal_dist(sample_1, sample_2, sample_3, v)
                if dist < min_dist:
                    min_dist = dist
                    min_sample_1 = sample_1
                    min_sample_2 = sample_2
                    min_sample_3 = sample_3
                    label = []
                    label.append(trainset_no_random[index[i]][1])
                    label.append(trainset_no_random[index[j]][1])
                    label.append(trainset_no_random[index[k]][1])
                    label_final = label

    return min_sample_1, min_sample_2, min_sample_3, label_final

def lasso_regression_find_plane(trainset_no_random, v):
    y= v.flatten().numpy()
    #x_final = trainset_no_random[len(trainset_no_random)-1][0].flatten().numpy()
    #y -= x_final
    flattened_vectors = []

    # Iterate through trainset_no_random and flatten each vector
    for i in range(len(trainset_no_random)):  
        flattened_vector = trainset_no_random[i][0].flatten().numpy() 
        flattened_vectors.append(flattened_vector)

    # Convert the list of flattened vectors to a NumPy array
    X = np.array(flattened_vectors)
    X = X.T
    
    lasso = Lasso(alpha=0.5)
    lasso.fit(X, y)

    # 输出所有系数的和
    print(f"Sum of coefficients: {np.sum(lasso.coef_)}")
    # 增加一个系数，为1-所有系数的和
    #lasso.coef_ = np.append(lasso.coef_, 1 - np.sum(lasso.coef_))
    
    print(f"Number of non-zero coefficients: {np.sum(lasso.coef_ != 0)}")
    
    # 取出非零的系数对应的样本点,从大到小排序
    index = np.argsort(-lasso.coef_)
   
    # 排序后重新输出所有非零系数
    print(f"Coefficients: {lasso.coef_[index]}")
    

    sample_1 = trainset_no_random[index[0]][0]
    sample_2 = trainset_no_random[index[1]][0]
    sample_3 = trainset_no_random[index[2]][0]
    label = []
    label.append(trainset_no_random[index[0]][1])
    label.append(trainset_no_random[index[1]][1])
    label.append(trainset_no_random[index[2]][1])

    return sample_1, sample_2, sample_3, label


def scs_find_plane(trainset_no_random, v):
    y= v.flatten().numpy()
    #x_final = trainset_no_random[len(trainset_no_random)-1][0].flatten().numpy()
    #y -= x_final
    flattened_vectors = []
    # Iterate through trainset_no_random and flatten each vector
    for i in range(len(trainset_no_random)):  
        flattened_vector = trainset_no_random[i][0].flatten().numpy() 
        flattened_vectors.append(flattened_vector)

    # Convert the list of flattened vectors to a NumPy array
    X = np.array(flattened_vectors)
    
    # 定义变量和参数  
    beta = cp.Variable(len(trainset_no_random))  
    lambda_ = 0.1  
    
    # 定义目标函数  
    objective = cp.Minimize(cp.norm2(y - beta@X) + lambda_*cp.norm1(beta))  
    
    # 定义约束条件  
    constraints = [cp.sum(beta) == 1]  
    
    # 定义问题  
    prob = cp.Problem(objective, constraints)  
    
    # 解决问题，使用SCS求解器，精度设为较低的值  
    prob.solve(solver=cp.SCS, eps_abs = 1e-2, verbose = True, use_indirect = False)  
    
    # 输出结果  
    print("The optimal value is", prob.value) 
    print("The optimal beta is", beta.value)
    
    # 取出非零的系数对应的样本点,从大到小排序
    index = np.argsort(-beta.value)

    sample_1 = trainset_no_random[index[0]][0]
    sample_2 = trainset_no_random[index[1]][0]
    sample_3 = trainset_no_random[index[2]][0]
    label = []
    label.append(trainset_no_random[index[0]][1])
    label.append(trainset_no_random[index[1]][1])
    label.append(trainset_no_random[index[2]][1])

    return sample_1, sample_2, sample_3, label

def near_scs_find_plane(trainset_no_random, v):
    y= v.flatten().numpy()
    #x_final = trainset_no_random[len(trainset_no_random)-1][0].flatten().numpy()
    #y -= x_final
    flattened_vectors = []
    dist = []
    for i in range(0, len(trainset_no_random)):
        dist.append(np.linalg.norm(trainset_no_random[i][0].flatten().numpy() - v.flatten().numpy()))
    
    dist = np.array(dist)
    dex = np.argsort(dist)
    # 取出离得最近的500个点
    num = 3
    dex = dex[1:num+1]
    # Iterate through trainset_no_random and flatten each vector
    for i in range(num):  
        flattened_vector = trainset_no_random[dex[i]][0].flatten().numpy() 
        flattened_vectors.append(flattened_vector)

    # Convert the list of flattened vectors to a NumPy array
    X = np.array(flattened_vectors)
    
    # 定义变量和参数  
    beta = cp.Variable(num)  
    lambda_ = 1  
    
    # 定义目标函数  
    objective = cp.Minimize(cp.norm2(y - beta@X) + lambda_*cp.norm1(beta))  
    
    # 定义约束条件  
    constraints = [cp.sum(beta) == 1]  
    
    # 定义问题  
    prob = cp.Problem(objective, constraints)  
    
    # 解决问题，使用OSQP求解器，精度设为较低的值  
    prob.solve(solver=cp.SCS, eps_abs = 1e-2,  use_indirect = False)  
    
    # 输出结果  
    print("The optimal value is", prob.value) 
    print(sum(beta.value))
    
    # 输出beta中非零元素的个数
    cnt = 0
    um = 0
    for i in range(num):
        if beta.value[i] > 1e-3:
            cnt += 1
        else:
            um += beta.value[i]
    print(um)  
    print("The number of non-zero beta is", cnt)
    #print("The optimal beta is", beta.value)
    
    # 取出非零的系数对应的样本点,从大到小排序
    index = np.argsort(-beta.value)
    print(index)
    print(beta.value[index[0]], beta.value[index[1]], beta.value[index[2]])
    sample_1 = trainset_no_random[dex[index[0]]][0]
    sample_2 = trainset_no_random[dex[index[1]]][0]
    sample_3 = trainset_no_random[dex[index[2]]][0]
    label = []
    label.append(trainset_no_random[dex[index[0]]][1])
    label.append(trainset_no_random[dex[index[1]]][1])
    label.append(trainset_no_random[dex[index[2]]][1])

    return sample_1, sample_2, sample_3, label