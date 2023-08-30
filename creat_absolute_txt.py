import os


def creat_txt(name):
    txt_path = os.path.join(path, name + '.txt')
    read_txt = os.path.join(read_path, name + '.txt')
    txt = open(txt_path, 'w')
    with open(read_txt, 'r') as f:
        for line in f:
            line = line.strip('\n')
            line = line + '.jpg\n'
            if name == 'train':
                txt.write(images_path + '/' + line)
            elif name == 'val':
                txt.write(images_path + '/' + line)
        txt.close()


if __name__ == '__main__':
    path = r'F:\Artificial_neural_Network\yolov5_Trackpad\train_data'
    read_path = r'F:\Artificial_neural_Network\yolov8-main\mydata_predict\dataSet'
    images_path = r'F:\Artificial_neural_Network\yolov5_Trackpad\train_data\images\val'
    list = ['train', 'val']
    for i in list:
        creat_txt(i)
