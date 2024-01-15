import os
import time
import random
    
random.seed(0)


os.chdir(os.path.dirname(os.getcwd()))

nets = ['ShuffleNetV2', 'EfficientNetB0', 'RegNetX_200MF', 'SimpleDLA']
# 'Resnet18','Resnet34', 'VGG19', 'MobileNet', 'SENet18', 'ShuffleNetV2', 'EfficientNetB0', 'RegNetX_200MF', 'SimpleDLA']

seeds = [0,1,2]

data_choose = [0]

learning_rate = [0.1,0.01,0.001]
batch_size = [1024,512,256]
weight_decay = [1e-5,1e-6,1e-7]

# for net in nets:
#     for data in data_choose:
#         for seed in seeds:
#             for lr in learning_rate:
#                 for batch in batch_size:
#                     for weight in weight_decay:
#                         os.system(f"python main.py --dir {'finalgraph/'+ str(net) + '_' + str(lr) + '_' + str(batch) + '_' + str(weight) + '_' + str(data) + '/' + str(seed) } --training_epochs 200 --weight_decay {weight} --lr {lr} --batch_size {batch} --skip_plot 10 --seed {seed} --net {net} --data_choose {data} --task mixup --mixup 1&")
#                         time.sleep(300)

# for net in nets:
#     for data in data_choose:
#         for seed in seeds:
#             for lr in learning_rate:
#                 for batch in batch_size:
#                     for weight in weight_decay:
#                         os.system(f"python main.py --dir {'finalgraph/'+ str(net) + '_' + str(lr) + '_' + str(batch) + '_' + str(weight) + '_' + str(data) + '/' + str(seed) } --training_epochs 200 --weight_decay {weight} --lr {lr} --batch_size {batch} --skip_plot 10 --seed {seed} --net {net} --data_choose {data} --task augmentation --random 1&")
#                         time.sleep(300)

optimizer = ['adam']

for net in nets:
    for data in data_choose:
        for seed in seeds:
            for opt in optimizer:
                for lr in learning_rate:
                    for batch in batch_size:
                        for weight in weight_decay:
                            os.system(f"python main.py --dir {'finalgraph/'+ str(net) + '_' + str(lr) + '_' + str(batch) + '_' + str(weight) + '_' + str(data) + '/' + str(seed) } --training_epochs 200 --optimizer {opt} --weight_decay {weight} --lr {lr} --batch_size {batch} --skip_plot 10 --seed {seed} --net {net} --data_choose {data} --task adam&")
                            time.sleep(500)


# learning_rate = [0.1,0.05,0.01,0.005,0.001]

# for net in nets:
#     for data in data_choose:
#         for seed in seeds:
#             for lr in learning_rate:
#                 os.system(f"python main.py --dir {'results_one/'+ str(net) + '_' + str(lr) + '_' + str(data) + '/' + str(seed) } --training_epochs 200 --lr {lr} --skip_plot 10 --seed {seed} --net {net} --data_choose {data} &")
#                 time.sleep(500)

# batch_size = [32,64,128,256,512,1024]
# for net in nets:
#     for data in data_choose:
#         for seed in seeds:
#             for batch in batch_size:
#                 os.system(f"python main.py --dir {'results_one/'+ str(net) + '_' + str(batch) + '_' + str(data) + '/' + str(seed) } --training_epochs 200 --batch_size {batch} --skip_plot 10 --seed {seed} --net {net} --data_choose {data} &")
#                 time.sleep(500)