import json
import os

def change_json_label_name(json_floder_path, json_name, label_pre, label_pro):

    json_path = os.path.join(json_floder_path, json_name)  # os路径融合
    data = json.load(open(json_path, 'r', encoding='gb2312', errors='ignore'))

    # len(data['shapes'])
    for i in data['shapes']:
        label_name = i['label']  # 得到json中你标记的类名
        if label_name == label_pre:
            i['label'] = label_pro