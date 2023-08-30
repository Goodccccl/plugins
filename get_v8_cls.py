import os
import random
import shutil

import torch

path = r'F:\Workprojects\TongFu_Bump\data\Manually_select_data2\normal'
# path = r'F:\Workprojects\TongFu_Bump\data\Manually_select_data2\error'
lists = os.listdir(path)
k = 0.8
num = len(lists)
train_num = int(num * k)
val_num = num - train_num
train_lists = random.sample(lists, train_num)
val_lists = []
for i in lists:
    if i not in train_lists:
        val_lists.append(i)

new_train_path = r"F:\Artificial_neural_Network\yolov8-main\mydata_classify\train\normal"
# new_train_path = r"F:\Artificial_neural_Network\yolov8-main\mydata_classify\train\error"

new_val_path = r"F:\Artificial_neural_Network\yolov8-main\mydata_classify\val\normal"
# new_val_path = r"F:\Artificial_neural_Network\yolov8-main\mydata_classify\val\error"

for name in train_lists:
    shutil.copy(path + '/' + name, new_train_path + '/' + name)

for name in val_lists:
    shutil.copy(path + '/' + name, new_val_path + '/' + name)

print('结束')