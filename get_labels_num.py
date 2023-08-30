# 处理labelme多边形矩阵的标注  json转化txt,提取点
import json
import os


def decode_json(json_floder_path, json_name):
        pengshang = 0
        glue_overflow = 0
        stains = 0
        scratch = 0
        json_path = os.path.join(json_floder_path, json_name)  # os路径融合
        data = json.load(open(json_path, 'r', encoding='gb2312', errors='ignore'))

        for i in data['shapes']:
            label_name = i['label']  # 得到json中你标记的类名
            if label_name == 'pengshang':
                pengshang += 1
            elif label_name == 'glue_overflow':
                glue_overflow += 1
            elif label_name == 'stains':
                stains += 1
            elif label_name == 'scratch':
                scratch += 1
        return pengshang, glue_overflow, stains, scratch





if __name__ == "__main__":
    json_floder_path = r'H:\six-xianchang\chen\side-label\json'  # 存放json的文件夹的绝对路径
    pengshang, glue_overflow, stains, scratch = 0, 0, 0, 0
    json_names = os.listdir(json_floder_path)
    print("共有：{}个文件待转化".format(len(json_names)))
    flagcount = 0
    for json_name in json_names:
        if json_name.split('.')[1] == 'json':
            pengshang1, glue_overflow1, stains1, scratch1 = decode_json(json_floder_path, json_name)
            pengshang = pengshang + pengshang1
            glue_overflow = glue_overflow + glue_overflow1
            stains = stains + stains1
            scratch = scratch + scratch1
            flagcount += 1
            print("还剩下{}个文件未转化".format(len(json_names) - flagcount))
        else:
            continue

    print('-------------查找完毕--------------')
    print('pengshang有{}个'.format([pengshang]))
    print('glue_overflow有{}个'.format([glue_overflow]))
    print('stains有{}个'.format([stains]))
    print('scratch有{}个'.format([scratch]))
