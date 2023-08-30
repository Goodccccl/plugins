# 处理labelme多边形矩阵的标注  json转化txt,提取点
import json
import os
import random

# name2id = {'bike': 0, 'arrow': 1, 'crossline': 2, 'building': 3, 'car': 4, 'person': 5}
name2id = {'BlackDot': 0, 'Birdging': 1, 'Stain': 2, 'Scratch': 3, 'Peeling': 4, 'Bump': 5}


def decode_json(json_floder_path, txt_outer_path, json_name):
    txt_name = txt_outer_path + json_name[:-5] + '.txt'
    with open(txt_name, 'a') as f:
        json_path = os.path.join(json_floder_path, json_name)  # os路径融合
        data = json.load(open(json_path, 'r', encoding='gb2312', errors='ignore'))
        img_w = data['imageWidth']  # 图片的高
        img_h = data['imageHeight']  # 图片的宽
        isshape_type = data['shapes'][0]['shape_type']
        print(isshape_type)
        dw = 1. / (img_w)
        dh = 1. / (img_h)
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


if __name__ == "__main__":
    json_floder_path = r'F:\new_g\json'  # 存放json的文件夹的绝对路径
    txt_outer_path = r'F:\1/'  # 存放txt的文件夹绝对路径
    json_names = os.listdir(json_floder_path)
    print("共有：{}个文件待转化".format(len(json_names)))
    flagcount = 0
    for json_name in json_names:
        if json_name.split('.')[1] == 'json':
            decode_json(json_floder_path, txt_outer_path, json_name)
            flagcount += 1
            print("还剩下{}个文件未转化".format(len(json_names) - flagcount))
        else:
            continue

    print('-------------转化全部完毕--------------')
