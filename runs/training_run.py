import os
import time
import random
    
random.seed(0)


os.chdir(os.path.dirname(os.getcwd()))

nets = ['SimpleDLA']
# 'resnet18','resnet34', 'resnet50', 'VGG19', 'MobileNet', 'SENet18', 'ShuffleNetV2', 'EfficientNetB0', 'RegNetX_200MF', 'SimpleDLA']

seeds = [0,1,2]

data_choose = [4]
# data_choose = [5,6,0]

learning_rate = [0.1,0.01,0.001]
batch_size = [256,512,1024]
weight_decay = [1e-5,1e-6,1e-7]

for net in nets:
    for data in data_choose:
        for seed in seeds:
            for lr in learning_rate:
                for batch in batch_size:
                    for weight in weight_decay:
                        os.system(f"python main.py --dir {'finalgraph/'+ str(net) + '_' + str(lr) + '_' + str(batch) + '_' + str(weight) + '_' + str(data) + '/' + str(seed) } --training_epochs 200 --weight_decay {weight} --lr {lr} --batch_size {batch} --skip_plot 10 --seed {seed} --net {net} --data_choose {data} &")
                        time.sleep(300)

#optimizer = ['sgd', 'adam','adagrad', 'rmsprop']
# optimizer = ['adam']

# for net in nets:
#     for data in data_choose:
#         for seed in seeds:
#             for opt in optimizer:
#                 for lr in learning_rate:
#                     for batch in batch_size:
#                         for weight in weight_decay:
#                             os.system(f"python main.py --dir {'finalgraph/'+ str(net) + '_' + str(lr) + '_' + str(batch) + '_' + str(weight) + '_' + str(data) + '/' + str(seed) } --training_epochs 200 --optimizer {opt} --weight_decay {weight} --lr {lr} --batch_size {batch} --skip_plot 10 --seed {seed} --net {net} --data_choose {data} &")
#                             time.sleep(1000)


# learning_rate = [0.1,0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01,0.009,0.008,0.007,0.006,0.005,0.004,0.003,0.002,0.001]

# for net in nets:
#     for data in data_choose:
#         for seed in seeds:
#             for lr in learning_rate:
#                 for dev in device:
#                     os.system(f"python main.py --dir {'results_one/'+ str(net) + '_' + str(lr) + '_' + str(data) + '/' + str(seed) } --training_epochs 200 --lr {lr} --skip_plot 10 --device {dev} --seed {seed} --net {net} --data_choose {data} &")
#                     time.sleep(1000)
#     time.sleep(6000)

# batch_size = [8,16,32,64,128,256,512,1024,2048]
# for net in nets:
#     for data in data_choose:
#         for seed in seeds:
#             for batch in batch_size:
#                 for dev in device:
#                     os.system(f"python main.py --dir {'results_one/'+ str(net) + '_' + str(batch) + '_' + str(data) + '/' + str(seed) } --training_epochs 200 --batch_size {batch} --skip_plot 10 --device {dev} --seed {seed} --net {net} --data_choose {data} &")
#                     time.sleep(1000)
#     time.sleep(6000)