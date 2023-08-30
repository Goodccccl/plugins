import random
import shutil
import os
import json
from copy import deepcopy

import cv2

from json2xml.create_xml_anno import CreateAnno
from json2xml.read_json_anno import ReadAnno
from Augment.augment.Data_enhancement.DataAugForObjectSegmentation.DataAugmentforLabelMe import ToolHelper, DataAugmentForObjectDetection


def json_transform_xml(json, xml_path, process_mode="rectangle"):
    json_anno = ReadAnno(json, process_mode=process_mode)
    width, height = json_anno.get_width_height()
    filename = json_anno.get_filename()
    coordis = json_anno.get_coordis()

    xml_anno = CreateAnno()
    xml_anno.add_filename(filename)
    xml_anno.add_pic_size(width_text_str=str(width), height_text_str=str(height), depth_text_str=str(3))
    for xmin, ymin, xmax, ymax, label in coordis:
        xml_anno.add_object(name_text_str=str(label),
                            xmin_text_str=str(int(xmin)),
                            ymin_text_str=str(int(ymin)),
                            xmax_text_str=str(int(xmax)),
                            ymax_text_str=str(int(ymax)))

    xml_anno.save_doc(xml_path)


def json2xml(root_json_dir, dataset_path, process_mode='rectangle'):
    xml_path = os.path.join(dataset_path, 'xml')
    jsons_list = os.listdir(root_json_dir)
    for i in range(len(jsons_list)):
        json_filename = jsons_list[i]
        if json_filename.split('.')[-1] == 'json':
            json_path = os.path.join(root_json_dir, json_filename)
            save_xml_path = os.path.join(xml_path, json_filename.replace(".json", ".xml"))
            json_transform_xml(json_path, save_xml_path, process_mode)  # labelme原数据的标注方式(矩形rectangle和多边形polygon)


def convert(img_size, box):
    dw = 1. / (img_size[0])
    dh = 1. / (img_size[1])
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = abs(box[2] - box[0])
    h = abs(box[3] - box[1])
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def get_name2id(classes):
    name2id = {}
    for i in range(len(classes)):
        name2id.setdefault('{}'.format(classes[i]), i)
    return name2id


def json2txt_rectangle(json_path, txt_output_path, json_name, classes):
    txt_name = txt_output_path + '/' + json_name[:-5] + '.txt'
    with open(txt_name, 'w') as f:
        json_path = os.path.join(json_path, json_name)  # os路径融合
        data = json.load(open(json_path, 'r', encoding='gb2312', errors='ignore'))
        img_w = data['imageWidth']  # 图片的高
        img_h = data['imageHeight']  # 图片的宽
        isshape_type = data['shapes'][0]['shape_type']
        print(isshape_type)
        # print(isshape_type)
        # print('下方判断根据这里的值可以设置为你自己的类型，我这里是polygon'多边形)
        # len(data['shapes'])
        name2id = get_name2id(classes)
        for i in data['shapes']:
            label_name = i['label']  # 得到json中你标记的类名
            if (i['shape_type'] == 'rectangle'):  # 为矩形不需要转换
                x1 = float(i['points'][0][0])
                y1 = float(i['points'][0][1])
                x2 = float(i['points'][1][0])
                y2 = float(i['points'][1][1])
                bb = (x1, y1, x2, y2)
            bbox = convert((img_w, img_h), bb)
            try:
                f.write(str(name2id[label_name]) + " " + " ".join([str(a) for a in bbox]) + '\n')
            except:
                pass
        f.close()


def json2txt_polygon(json_path, txt_output_path, json_name, classes):
    txt_name = txt_output_path + '/' + json_name[:-5] + '.txt'
    with open(txt_name, 'a') as f:
        json_path = os.path.join(json_path, json_name)  # os路径融合
        data = json.load(open(json_path, 'r', encoding='gb2312', errors='ignore'))
        img_w = data['imageWidth']  # 图片的高
        img_h = data['imageHeight']  # 图片的宽
        isshape_type = data['shapes'][0]['shape_type']
        print(isshape_type)
        dw = 1. / (img_w)
        dh = 1. / (img_h)
        name2id = get_name2id(classes)
        for i in data['shapes']:
            label_name = i['label']  # 得到json中你标记的类名
            if (i['shape_type'] == 'polygon'):
                point = []
                for lk in range(len(i['points'])):
                    x = float(i['points'][lk][0])
                    y = float(i['points'][lk][1])
                    point_x = x * dw
                    point_y = y * dh
                    point.append(point_x)
                    point.append(point_y)
                f.write(str(name2id[label_name]) + " " + " ".join([str(a) for a in point]) + '\n')
        f.close()


def creat_ImageSet(dataset_path, train_p):
    img_path = os.path.join(dataset_path, 'images')
    name_lists = os.listdir(img_path)
    num = len(name_lists)
    list = range(num)
    tv_n = int(num * 1)
    train_n = int(num * train_p)
    trainval = random.sample(list, tv_n)
    train = random.sample(list, train_n)
    ftrainval = open(os.path.join(dataset_path,'dataSet', 'trainval.txt'), 'w')
    ftest = open(os.path.join(dataset_path,'dataSet', 'test.txt'), 'w')
    ftrain = open(os.path.join(dataset_path,'dataSet', 'train.txt'), 'w')
    fval = open(os.path.join(dataset_path,'dataSet', 'val.txt'), 'w')
    for i in list:
        name = name_lists[i].split('.')[0] + '\n'
        if i in trainval:
            ftrainval.write(name)
            if i in train:
                ftrain.write(name)
            else:
                fval.write(name)
        else:
            ftest.write(name)
    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()


def get_images(task, image_path, json_path, dataset_path, suffix):
    if task == 'predict':
        dataset_images_Dir = os.path.join(dataset_path, 'images')
        json_names = os.listdir(json_path)
        # image_names = glob.glob(os.path.join(image_path, '*.{}'.format(suffix)))
        for i in range(len(json_names)):
            image_name = json_names[i][:-5] + '.' + suffix
            shutil.copy(image_path + '/' + image_name, dataset_images_Dir + '/' + image_name)
    if task == 'segment':
        dataset_images_Dir = os.path.join(dataset_path, 'images', 'train2017')
        json_names = os.listdir(json_path)
        # image_names = glob.glob(os.path.join(image_path, '*.{}'.format(suffix)))
        for i in range(len(json_names)):
            image_name = json_names[i][:-5] + '.' + suffix
            shutil.copy(image_path + '/' + image_name, dataset_images_Dir + '/' + image_name)


def get_cls_classes(src_Dir):
    classes = os.listdir(src_Dir)
    return classes


def get_images_cls(image_Dirs, dataset_path, train_percent):
    dataset_images_t = os.path.join(dataset_path, 'train')
    dataset_images_v = os.path.join(dataset_path, 'val')
    classes = get_cls_classes(image_Dirs)
    for i in range(len(classes)):
        now_class = classes[i].strip()
        now_class_dir_t = os.path.join(dataset_images_t, now_class)
        now_class_dir_v = os.path.join(dataset_images_v, now_class)
        os.mkdir(now_class_dir_t)
        os.mkdir(now_class_dir_v)
        images_path = os.path.join(image_Dirs, now_class)
        names = os.listdir(images_path)
        num = len(names)
        list = range(num)
        train_n = int(num * train_percent)
        train = random.sample(list, train_n)
        for n in list:
            image_name = names[n].strip()
            if n in train:
                shutil.copy(images_path + '/' + image_name, now_class_dir_t + '/' + image_name)
            else:
                shutil.copy(images_path + '/' + image_name, now_class_dir_v + '/' + image_name)


def creat_labels(task, json_dir, dataset_path, shape_type, classes):
    json_names = os.listdir(json_dir)
    if task == 'predict':
        labels_path = os.path.join(dataset_path, 'labels')
        if shape_type == 'rectangle':
            for json_name in json_names:
                if json_name.split('.')[1] == 'json':
                    json2txt_rectangle(json_dir, labels_path, json_name, classes)

        if shape_type == 'polygon':
            for json_name in json_names:
                if json_name.split('.')[1] == 'json':
                    json2txt_polygon(json_dir, labels_path, json_name, classes)
    else:
        labels_path = os.path.join(dataset_path, 'labels', 'train2017')
        if shape_type == 'rectangle':
            for json_name in json_names:
                if json_name.split('.')[1] == 'json':
                    json2txt_rectangle(json_dir, labels_path, json_name, classes)

        if shape_type == 'polygon':
            for json_name in json_names:
                if json_name.split('.')[1] == 'json':
                    json2txt_polygon(json_dir, labels_path, json_name, classes)


def creat_absolute_txt(dataset_path, ImageSet_path, suffix):
    images_path = os.path.join(dataset_path, 'images')
    train_txt_r = os.path.join(ImageSet_path, 'train.txt')
    val_txt_r = os.path.join(ImageSet_path, 'val.txt')
    train_txt_ab_p = os.path.join(dataset_path, 'train.txt')
    val_txt_ab_p = os.path.join(dataset_path, 'val.txt')
    train_txt_ab = open(train_txt_ab_p, 'w')
    for line in open(train_txt_r):
        line = line.strip('\n')
        line = line + '.{}\n'.format(suffix)
        train_txt_ab.write(images_path + '/' + line)
    val_txt_ab = open(val_txt_ab_p, 'w')
    for line in open(val_txt_r):
        line = line.strip('\n')
        line = line + '.{}\n'.format(suffix)
        val_txt_ab.write(images_path + '/' + line)


def make_mydata_predict(task, src_image_dir, dataset_path, classes, train_percent, shape_type, suffix):
    json_dir = os.path.join(src_image_dir, 'json')
    get_images(task, src_image_dir, json_dir, dataset_path, suffix)
    creat_labels(task, json_dir, dataset_path, shape_type, classes)
    json2xml(json_dir, dataset_path, process_mode=shape_type)
    creat_ImageSet(dataset_path, train_percent)
    ImageSet_path = os.path.join(dataset_path, 'dataSet')
    creat_absolute_txt(dataset_path, ImageSet_path, suffix)


def make_dataset_dir_predict(dataset_path):
    dataSet = os.path.join(dataset_path, 'dataSet')
    images = os.path.join(dataset_path, 'images')
    labels = os.path.join(dataset_path, 'labels')
    xml = os.path.join(dataset_path, 'xml')
    os.mkdir(dataSet)
    os.mkdir(images)
    os.mkdir(labels)
    os.mkdir(xml)


def make_dataset_dir_classify(cls_src_dir, dataset_path, train_percent=0.8):
    train = os.path.join(dataset_path, 'train')
    val = os.path.join(dataset_path, 'val')
    os.mkdir(train)
    os.mkdir(val)
    # 取图操作
    get_images_cls(cls_src_dir, dataset_path, train_percent)


def make_dataset_dir_seg(task, src_image_dir, dataset_path, classes, shape_type, suffix):
    images = os.path.join(dataset_path, 'images')
    labels = os.path.join(dataset_path, 'labels')
    os.mkdir(images)
    os.mkdir(labels)
    images_train = os.path.join(images, 'train2017')
    labels_train = os.path.join(labels, 'train2017')
    os.mkdir(images_train)
    os.mkdir(labels_train)
    # 获取polygon多边形点位操作
    json_dir = os.path.join(src_image_dir, 'json')
    creat_labels(task, json_dir, dataset_path, shape_type, classes)
    get_images(task, src_image_dir, json_dir, dataset_path, suffix)


def Augment_json(src_image_dir, aug_times):
    toolhelper = ToolHelper()
    dataAug = DataAugmentForObjectDetection()
    json_dir = os.path.join(src_image_dir, 'json')
    json_lists = os.listdir(json_dir)
    for i in json_lists:
        cnt = 0
        json_name = i.strip()
        json_path = os.path.join(json_dir, json_name)
        image_name = json_name.split('.')[0] + '.' + suffix
        image_path = os.path.join(src_image_dir, image_name)
        json_dic = toolhelper.parse_json(json_path)
        img = cv2.imread(image_path)
        img_prefix = image_name.split('.')[0]
        img_suffix = image_name.split('.')[1]
        while cnt < aug_times:
            auged_img, json_info = dataAug.dataAugment(deepcopy(img), deepcopy(json_dic))
            img_name = '{}_{}.{}'.format(img_prefix, cnt + 1, img_suffix)
            img_save_path = os.path.join(src_image_dir, img_name)
            toolhelper.save_img(img_save_path, auged_img)

            json_info['imagePath'] = img_name
            base64_data = toolhelper.img2str(img_save_path)
            json_info['imageData'] = base64_data
            toolhelper.save_json('{}_{}.json'.format(img_prefix, cnt + 1), json_dir, json_info)
            print(img_name)
            cnt += 1
    print("————————————————————————————————增强完毕————————————————————————————————")



if __name__ == "__main__":
    # 原始数据放置， 图片文件夹下创建json文件夹存放所有的json文件即可
    task = 'predict'                    # 任务名 predict, classify, segment
    augment = True                     # 是否启用数据增强
    aug_times = 3                       # 每张图增强次数
    src_image_dir = r'F:\test'           # 输入原始数据文件夹地址
    dataset_path = r'F:\test\yolov8_dataset'           # 输出数据集保存地址
    classes = ['scratch', 'qipao']         # 类别名
    train_percent = 0.8                 # 训练和验证占比
    shape_type = 'rectangle'              # json中标签是多边形（polygon）还是四边形（rectangle）
    suffix = 'bmp'                      # 原始数据图片后缀
    sz = os.listdir(dataset_path)
    if augment == True:
        Augment_json(src_image_dir, aug_times)
        if not sz:
            print("数据集存放文件夹为空，开始制作{}数据集！".format(task))
            if task == 'predict':
                make_dataset_dir_predict(dataset_path)
                make_mydata_predict(task, src_image_dir, dataset_path, classes, train_percent, shape_type, suffix)
            if task == 'segment':
                make_dataset_dir_seg(task, src_image_dir, dataset_path, classes, shape_type, suffix)
            if task == 'classify':
                make_dataset_dir_classify(src_image_dir, dataset_path, train_percent)

        else:
            print("数据集存放文件夹非空，请清空或者更换文件夹！")
    else:
        if not sz:
            print("数据集存放文件夹为空，开始制作{}数据集！".format(task))
            if task == 'predict':
                make_dataset_dir_predict(dataset_path)
                make_mydata_predict(task, src_image_dir, dataset_path, classes, train_percent, shape_type, suffix)
            if task == 'segment':
                make_dataset_dir_seg(task, src_image_dir, dataset_path, classes, shape_type, suffix)
            if task == 'classify':
                make_dataset_dir_classify(src_image_dir, dataset_path, train_percent)

        else:
            print("数据集存放文件夹非空，请清空或者更换文件夹！")
