import os
import shutil


# 这个库复制文件比较省事

def objFileName(json_path):
    '''
    生成文件名列表
    :return:
    '''
    local_file_name_list = os.listdir(json_path)
    # 指定名单
    obj_name_list = []
    for i in local_file_name_list:
        obj_name_list.append(i.replace('\n', ''))
    return obj_name_list



def copy_img():
    '''
    复制、重命名、粘贴文件
    :return:
    '''
    json_path = r'H:\six-xianchang\chen\side-label\break_json'
    # 指定要复制的图片路径
    local_img_name = r'H:\six-xianchang\chen\side-label'

    # 指定存放图片的目录
    path = r'H:\six-xianchang\chen\side-label\break_json'

    for i in objFileName(json_path):
        new_obj_name = i[:-5] + '.bmp'
        shutil.copy(local_img_name + '/' + new_obj_name, path + '/' + new_obj_name)

def copy_txt():
    img_names_path = r"F:\Artificial_neural_Network\yolov5_Trackpad\train_data\images\train"
    local_txt_path = r"F:\Workprojects\wafer3dproject-developer\LabelImg\txt"
    path = r"F:\Artificial_neural_Network\yolov5_Trackpad\train_data\labels\train"
    for i in objFileName(img_names_path):
        new_obj_name = i[:-4] + '.txt'
        shutil.copy(local_txt_path + '/' + new_obj_name, path + '/' + new_obj_name)

if __name__ == '__main__':
    copy_img()
    # copy_txt()