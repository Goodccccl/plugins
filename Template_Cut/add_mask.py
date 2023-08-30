import glob
import os
import time

import cv2
import numpy as np
import yaml
import torch

def add_mask(image_path, HotMap_path):
    image = cv2.imread(image_path)
    image1 = cv2.imread(HotMap_path, 0)

    mask = np.zeros((image.shape[0], image.shape[1], image.shape[2]))
    for i in range(452):
        for j in range(452):
            rgb = image1[i][j]
            if rgb >= 115:
                mask[i][j] = [0, 0, 255]
    mask = mask.astype(np.uint8)
    result = cv2.addWeighted(image, 0.6, mask, 0.4, 0)
    return result




if __name__ == "__main__":
    images_path = r"G:\estimate_811\membrane_cut"
    HotMaps_path = r"G:\estimate_811\padim_resnet18\Hot"
    result_save_dir = r"G:\estimate_811\padim_resnet18\masks_115"
    if not os.path.exists(result_save_dir):
        os.mkdir(result_save_dir)

    lists = os.listdir(HotMaps_path)
    for image_name in lists:
        if image_name[-4:] == ".bmp":
            image_path = os.path.join(images_path, image_name)
            HotMap_path = os.path.join(HotMaps_path, image_name)
            add_result = add_mask(image_path, HotMap_path)
            save_path = os.path.join(result_save_dir, image_name)
            cv2.imwrite(save_path, add_result)
            print("————————————{}：生成完毕————————————".format(image_name))



