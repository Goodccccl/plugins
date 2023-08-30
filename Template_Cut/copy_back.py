import os

import cv2
import numpy as np

if __name__ == "__main__":
    image1_dir = r"G:\estimate_811\membrane"
    image2_dir = r"G:\estimate_811\patchcore_resnet18\Hot"
    txt_dir = r"G:\estimate_811\txt"
    save_dir = r"G:\estimate_811\together_patchcore_Hot"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    txts = os.listdir(txt_dir)
    for txt in txts:
        image1_name = txt[0:-4] + '.bmp'
        image1_path = os.path.join(image1_dir, image1_name)
        image1 = cv2.imread(image1_path, 1)
        result_image = image1.copy()
        txt_path = os.path.join(txt_dir, txt)
        with open(txt_path, 'r') as r:
            context = r.readlines()
            num = len(context)
            for n in range(num):
                image2_name = txt[0:-4] + "_" + str(n) + '.bmp'
                xyxy = context[n].replace('\n', '').split(' ')
                image2_path = os.path.join(image2_dir, image2_name)
                image2 = cv2.imread(image2_path, 1)
                result_image[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])] = image2
        save_path = os.path.join(save_dir, image1_name)
        cv2.imwrite(save_path, result_image)