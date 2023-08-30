import os


def creat_txt(name):
    txt_path = os.path.join(path, name + '.txt')
    txt = open(txt_path, 'w')
    if name == 'train':
        name_list = os.listdir(train_images_path)
        for line in name_list:
            line = line.strip('\n')
            # line = line + '.tif\n'
            if name == 'train':
                txt.write(train_images_path + '/' + line + '\n')
        txt.close()
    if name == 'val':
        name_list = os.listdir(val_images_path)
        for line in name_list:
            line = line.strip('\n')
            # line = line + '.tif\n'
            if name == 'val':
                txt.write(val_images_path + '/' + line + '\n')
        txt.close()



if __name__ == '__main__':
    path = r'F:\Artificial_neural_Network\yolov5_Trackpad\train_data'
    train_images_path = r'F:\Artificial_neural_Network\yolov5_Trackpad\train_data\images\train'
    val_images_path = r'F:\Artificial_neural_Network\yolov5_Trackpad\train_data\images\val'
    list = ['train', 'val']
    for i in list:
        creat_txt(i)
