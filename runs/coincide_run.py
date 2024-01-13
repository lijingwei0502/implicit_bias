import os
import time


# p = [[2,18],[20,180],[1.25,1.25],[800,1200],[0.2,0.2]]
# step = [0.5,5,0.05,20,0.02]
# p0 = [10,100,1.25,1000,0.2]
# p1 = [10,100,1.25,1000,0.2]

# arg_lists=[]
# for i in range(0,5):
#     for j in range(int((p[i][1]-p[i][0])/step[i])+1):
#         p0[i]=p[i][0]+step[i]*j
#         arg_lists.append(p0.copy())
#     p0[i] = p1[i]
    
# os.chdir(os.path.dirname(os.getcwd()))
# for arg_list in arg_lists:
#     os.system(f"python -W ignore train_dense.py --env_name half_cheetah --seed 0 --frame_stack 1 --dr=false --max_steps 500000 --arg1 {arg_list[0]} --arg2 {arg_list[1]} --arg3 {arg_list[2]} --arg4 {arg_list[3]} --arg5 {arg_list[4]} --eval_only=True&")
#     time.sleep(10) 

point_nums = [2,3,4,5,6,7,8,9,10]
count_numbers = [10,50,100,1000,5000]
seeds = [0,1,2,3,4,5,6,7,8,9]
for seed in seeds:
    for point_num in point_nums:
        for count_number in count_numbers:
            os.system(f"python main.py -r --dir another_model --generate_point_num {point_num} --count_numbers {count_number} --seed {seed} --cal_pro_coincide True&")
            time.sleep(30) 
