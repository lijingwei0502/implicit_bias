import os
import time
import random
    
random.seed(0)

#nets = ['Resnet34','VGG19','MobileNet','SENet18']

nets = ['Resnet18']
seeds = [0]

data_choose = [1]
learning_rate = [0.1,0.01,0.001]
batch_size = [256,512,1024]
weight_decay = [1e-5,1e-6,1e-7]

for net in nets:
    for data in data_choose:
        for seed in seeds:
            for lr in learning_rate:
                for batch in batch_size:
                    for wd in weight_decay:
                        os.system(f"python main.py --dataset cifar100 --dir {'finalgraph/'+ str(net) + '_' + str(lr) + '_' + str(batch) + '_' + str(wd)} --training_epochs 200 --lr {lr} --batch_size {batch} --skip_plot 1 --seed {seed} --net {net} --data_choose {data} --weight_decay {wd} &")
                        time.sleep(15)
# scope_s = [[-1,2],[-2,3],[-3,4],[-4,5],[-5,6]]
# for net in nets:
#     for data in data_choose:
#         for seed in seeds:
#             for lr in learning_rate:
#                 for batch in batch_size:
#                     for wd in weight_decay:
#                         for scope_l, scope_r in scope_s:
#                             os.system(f"python main.py --dir {'finalgraph/'+ str(net) + '_' + str(lr) + '_' + str(batch) + '_' + str(wd) + '_' + str(data) + '/' + str(seed) } --training_epochs 200 --lr {lr} --batch_size {batch} --skip_plot 10 --seed {seed} --net {net} --data_choose {data} --weight_decay {wd} --scope_l {scope_l} --scope_r {scope_r}&")
#                             time.sleep(200)


data_choose = [1]

# ml = [1]
# kk = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
# pp = [1,2]
# weights = ['uniform','distance']
# for m in ml:
#     for data in data_choose:
#         for seed in seeds:
#             for k in kk:
#                 for p in pp:
#                     for w in weights:
#                         os.system(f"python main.py --dir {'finalgraph/ml/' + str(m) + '/' + str(data) + '/' + str(seed) } --training_epochs 200 --lr 0.01 --batch_size 256 --skip_plot 10 --seed {seed} --net Resnet18 --data_choose {data} --machine_learning {m} --k {k} --p {p} --weights {w}&")
#                         time.sleep(300)

# ml = [3]
# depths = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
# criterions = ['gini','entropy']
# splitter = ['best','random']

# for m in ml:
#     for data in data_choose:
#         for seed in seeds:
#             for d in depths:
#                 for c in criterions:
#                     for s in splitter:
#                         os.system(f"python main.py --dir {'finalgraph/ml/' + str(m) + '/' + str(data) + '/' + str(seed) } --training_epochs 200 --lr 0.01 --batch_size 256 --skip_plot 10 --seed {seed} --net Resnet18 --data_choose {data} --machine_learning {m} --max_depth {d} --criterion {c} --splitter {s}&")
#                         time.sleep(300)

# ml = [2]
# kernels = ['linear','poly','rbf', 'sigmoid', 'precomputed']
# Cs = [0.1,1,10,100]
# for m in ml:
#     for data in data_choose:
#         for seed in seeds:
#             for k in kernels:
#                 for c in Cs:
#                     os.system(f"python main.py --dir {'finalgraph/ml/' + str(m) + '/' + str(data) + '/' + str(seed) } --training_epochs 200 --lr 0.01 --batch_size 256 --skip_plot 10 --seed {seed} --net Resnet18 --data_choose {data} --machine_learning {m} --kernel {k} --C {c}&")
#                     time.sleep(300)

# ml = [3]
# n_neighbors = [20,50,100,150,200,250,300]
# depths = [3,4,5,7,10,12,15]
# criterions = ['gini','entropy']


# for m in ml:
#     for data in data_choose:
#         for seed in seeds:
#             for d in depths:
#                 for c in criterions:
#                     for s in splitter:
#                         os.system(f"python main.py --dir {'finalgraph/ml/' + str(m) + '/' + str(data) + '/' + str(seed) } --training_epochs 200 --lr 0.01 --batch_size 256 --skip_plot 10 --seed {seed} --net Resnet18 --data_choose {data} --machine_learning {m} --max_depth {d} --criterion {c} --splitter {s}&")
#                         time.sleep(300)

# ml = [4]
# Cs = [0.1,0.5,1,10,20,30,50,100]
# penalties = ['l1','l2','elasticnet','none']
# for m in ml:
#     for data in data_choose:
#         for seed in seeds:
#             for c in Cs:
#                 for p in penalties:
#                     os.system(f"python main.py --dir {'finalgraph/ml/' + str(m) + '/' + str(data) + '/' + str(seed) } --training_epochs 200 --lr 0.01 --batch_size 256 --skip_plot 10 --seed {seed} --net Resnet18 --data_choose {data} --machine_learning {m} --C {c} --penalty {p}&")
#                     time.sleep(300)

nets = ['Resnet18','Resnet34','SimpleDLA']
# 'Resnet18','Resnet34','SimpleDLA','VGG19','EfficientNetB0','RegNetX_200MF', 'MobileNet', 'SENet18', 'EfficientNetB0'
# learning_rate = [0.1,0.01,0.001]
# batch_size = [1024,512,256]
# weight_decay = [1e-5,1e-6,1e-7]
# data_choose = [3,4,5,6]

# for net in nets:
#     for data in data_choose:
#         for seed in seeds:
#             for lr in learning_rate:
#                 for batch in batch_size:
#                     for wd in weight_decay:
#                         os.system(f"python main.py --dir {'finalgraph/'+ str(net) + '_' + str(lr) + '_' + str(batch) + '_' + str(wd) + '_' + str(data) + '/' + str(seed) } --training_epochs 200 --lr {lr} --batch_size {batch} --skip_plot 10 --seed {seed} --net {net} --data_choose {data} --weight_decay {wd}&")
#                         time.sleep(300)

# for net in nets:
#     for data in data_choose:
#         for seed in seeds:
#             for lr in learning_rate:
#                 for batch in batch_size:
#                     for wd in weight_decay:
#                         os.system(f"python main.py --dir {'finalgraph/'+ str(net) + '_' + str(lr) + '_' + str(batch) + '_' + str(wd) + '_' + str(data) + '/' + str(seed) } --training_epochs 100 --lr {lr} --batch_size {batch} --skip_plot 10 --seed {seed} --net {net} --data_choose {data} --weight_decay {wd}&")
#                         time.sleep(100)

# for net in nets:
#     for data in data_choose:
#         for seed in seeds:
#             for lr in learning_rate:
#                 for batch in batch_size:
#                     os.system(f"python main.py --dir {'finalgraph/'+ str(net) + '_' + str(lr) + '_' + str(batch) + '_' + str(data) + '/' + str(seed) } --training_epochs 200 --lr {lr} --batch_size {batch} --skip_plot 10 --seed {seed} --net {net} --data_choose {data}&")
#                     time.sleep(500)