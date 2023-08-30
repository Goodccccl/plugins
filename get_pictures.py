# -*- coding: utf-8 -*-

import shutil


# 这个库复制文件比较省事

def objFileName():
    '''
    生成文件名列表
    :return:
    '''
    local_file_name_list = r'F:\Artificial_neural_Network\yolov8-main\mydata_predict\dataSet\val.txt'
    # 指定名单
    obj_name_list = []
    for i in open(local_file_name_list, 'r'):
        obj_name_list.append(i.replace('\n', ''))
    return obj_name_list


def copy_img():
    '''
    复制、重命名、粘贴文件
    :return:
    '''
    # 指定要复制的图片路径
    local_img_name = r'F:\Workprojects\wireless_charging\front_corner_drop\voc2012\JPEGImages'
    # local_img_name = r'D:\Workprojects\WireBonding_Meta\1031左右\VOC2012\SegmentationClassPNG'

    # 指定存放图片的目录
    path = r'F:\Artificial_neural_Network\yolov5_Trackpad\train_data\images\val'
    # path = r'D:\Artificial neural Network\yolov5ds-main\yolov5ds-main\yolov5\seg\labels\train'

    for i in objFileName():
        new_obj_name = i + '.jpg'
        shutil.copy(local_img_name + '/' + new_obj_name, path + '/' + new_obj_name)

# 复制标签txt文本文件，用于分文件夹后加入txt文本文件
def copy_txt():
    local_txt_name = r'F:\Workprojects\wireless_charging\front_corner_drop\labels\detect'

    path = r'F:\Artificial_neural_Network\yolov5_Trackpad\train_data\labels\val'

    for i in objFileName():
        new_obj_name = i + '.txt'
        shutil.copy(local_txt_name + '/' + new_obj_name, path + '/' + new_obj_name)

if __name__ == '__main__':
    copy_img()
    # copy_txt()