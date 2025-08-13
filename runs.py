import os
import time
import random
    
random.seed(0)

#nets = ['Resnet18','Resnet34','VGG19','MobileNet','SENet18']

nets = ['ShuffleNetV2','EfficientNetB0','RegNetX_200MF','SimpleDLA']
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
                        os.system(f"python main.py --dataset cifar100 --training_epochs 200 --lr {lr} --batch_size {batch} --skip_plot 10 --seed {seed} --net {net} --data_choose {data} --weight_decay {wd} &")
                        time.sleep(60)
