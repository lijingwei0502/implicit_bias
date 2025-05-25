import os
import time
import random
    
random.seed(0)


seeds = [1,2]

data_choose = [1]
learning_rate = [1e-3,1e-4,5e-5,1e-5]
batch_size = [1024,512,256]
weight_decay = [1e-5,1e-6,1e-7]
# 1e-6 1e-7

for data in data_choose:
    for seed in seeds:
        for lr in learning_rate:
            for batch in batch_size:
                for wd in weight_decay:
                    os.system(f"python main.py --lr {lr} --batch-size {batch} --seed {seed}  --data-choose {data} --weight-decay {wd}&")
                    time.sleep(300)
