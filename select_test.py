import os
import random

import numpy as np
import shutil

xml_path = r'F:\Artificial_neural_Network\yolov8-main\mydata\xml'
images_path = r'F:\Artificial_neural_Network\yolov8-main\mydata\images'
test_path = r'F:\Workprojects\front_corner_drop\test'

file_list = os.listdir(xml_path)
name_list = []
for i in range(len(file_list)):
    name = os.path.splitext(file_list[i])[0]
    name_list.append(name)
# print(name_list)

num = 100
high = len(name_list)
# print(high)
a = np.arange(0, high, 1)
a = list(a)
index = random.sample(a, num)
print(index)
select_name_list = []
for i in index:
    xml_n = name_list[i] + '.xml'
    image_n = name_list[i] + '.jpg'
    xml = os.path.join(xml_path, xml_n)
    image = os.path.join(images_path, image_n)
    shutil.move(image, test_path)
    shutil.move(xml, test_path)
print('--------------测试集选择完毕--------------')