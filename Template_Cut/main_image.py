import os
import cv2
import time
import numpy as np


def py_nms(dets, thresh):  # （模板匹配得到的符合范围的矩阵值，抑制的阈值）
    """Pure Python NMS baseline."""
    # x1、y1、x2、y2、以及score赋值
    # （x1、y1）（x2、y2）为box的左上和右下角标
    x1 = dets[:, 0]
    y1 = dets[:, 1]  # 左上角的坐标值

    x2 = dets[:, 2]
    y2 = dets[:, 3]  # 右下角的阈值

    scores = dets[:, 4]
    # 每一个候选框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # order是按照score降序排序的，从大到小
    order = scores.argsort()[::-1]
    # print("order:",order)

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        # 计算当前概率最大矩形框与其他矩形框的相交框的坐标，会用到numpy的broadcast机制，得到的是向量
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        # 计算相交框的面积,注意矩形框不相交时w或h算出来会是负数，用0代替
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # 计算重叠度IOU：重叠面积/（面积1+面积2-重叠面积）
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # 找到重叠度不高于阈值的矩形框索引
        inds = np.where(ovr <= thresh)[0]
        # print("inds:",inds)
        # 将order序列更新，由于前面得到的矩形框索引要比矩形框在原order序列中的索引小1，所以要把这个1加回来
        order = order[inds + 1]
    return keep

def template(img_gray, template_img, template_threshold):
    '''
    img_gray:待检测的灰度图片格式
    template_img:模板小图，也是灰度化了
    template_threshold:模板匹配的置信度
    '''

    h, w = template_img.shape[:2]  # 获取模板的高和宽
    res = cv2.matchTemplate(img_gray, template_img, cv2.TM_CCOEFF_NORMED)  # 模板匹配的方式
    start_time = time.time()
    loc = np.where(res >= template_threshold)  # 大于模板阈值的目标坐标，返回的就是矩阵的行列索引值，其中行坐标为坐标的y值，列坐标为x值
    score = res[res >= template_threshold]  # 大于模板阈值的目标置信度cv2.matchTemplate(img_gray, template_img, cv2.TM_CCOEFF_NORMED)[res >= template_threshold]完整写法
    # 将模板数据坐标进行处理成左上角、右下角的格式
    xmin = np.array(loc[1])  # 列坐标为x值
    ymin = np.array(loc[0])  # 横坐标为y值
    xmax = xmin + w
    ymax = ymin + h

    xmin = xmin.reshape(-1, 1)  # 变成n行1列维度
    xmax = xmax.reshape(-1, 1)  # 变成n行1列维度
    ymax = ymax.reshape(-1, 1)  # 变成n行1列维度
    ymin = ymin.reshape(-1, 1)  # 变成n行1列维度
    score = score.reshape(-1, 1)  # 变成n行1列维度

    data_hlist = []
    data_hlist.append(xmin)
    data_hlist.append(ymin)
    data_hlist.append(xmax)
    data_hlist.append(ymax)
    data_hlist.append(score)
    data_hstack = np.hstack \
        (data_hlist)  # 将xmin、ymin、xmax、yamx、scores按照列进行拼接       np.hstack():在水平方向上平铺  np.vstack():在竖直方向上堆叠
    thresh = 0.3  # NMS里面的IOU交互比阈值

    keep_dets = py_nms(data_hstack, thresh)  # 进行非极大值抑制
    print("nms time:", time.time() - start_time)  # 打印数据处理到nms运行时间
    dets = data_hstack[keep_dets]  # 最终的nms获得的矩形框
    return dets

def saveresult(name, file, img_rgb):
    isExists = os.path.exists("./ " + str(name) + '/')
    if not isExists:  # 判断如果文件不存在,则创建
        os.makedirs("./ " + str(name) + '/')
        print("./ " + str(name) + '/ ' + "目录创建成功")

    out = str(file.split('.')[0])
    filename = "./" + name + '/%s.jpg' % (out)
    print('已匹配图片：' + filename)
    cv2.imwrite(filename, img_rgb)


def savecut(save_path, file, cut, score):
    isExists = os.path.exists(save_path + '/')
    if not isExists:  # 判断如果文件不存在,则创建
        os.makedirs(save_path + '/')
        print(save_path + "目录创建成功")

    out = str(file.split('.')[0])
    filename = save_path + '/%s.bmp' % (out)
    print('已裁切图片：' + filename + ' ' + str(score))
    cv2.imwrite(filename, cut)

if __name__ == "__main__":
    images_dir = r"G:\estimate_811\membrane"
    save_path = r"G:\estimate_811\membrane_cut_480"
    os.makedirs(save_path)
    images_list = os.listdir(images_dir)
    for i in range(len(images_list)):
        image_name = images_list[i]
        im_path = os.path.join(images_dir, image_name)
        im = cv2.imread(im_path, 1)
        # cv2.namedWindow('im', cv2.WINDOW_NORMAL)
        # cv2.imshow('im', im)
        # cv2.waitKey(0)

        _, im2 = cv2.threshold(im, 200, 255, cv2.THRESH_BINARY)
        # cv2.namedWindow('im2', cv2.WINDOW_NORMAL)
        # cv2.imshow('im2', im2)
        # cv2.waitKey(0)

        template_path = r'G:\estimate_811\template_img.bmp'
        template_img = cv2.imread(template_path)
        _, template_img = cv2.threshold(template_img, 200, 255, cv2.THRESH_BINARY)
        # cv2.imshow('template_img', template_img)
        # cv2.waitKey(0)

        template_threshold = 0.5  # 模板置信度
        dets = template(im2, template_img, template_threshold)
        for j in range(dets.shape[0]):
            coord = dets[j]
            with open(r"{}\{}.txt".format(save_path, image_name[:-4]), 'a', encoding="utf-8") as w:
                x1 = str(int(coord[0]))
                y1 = str(int(coord[1]))
                x2 = str(int(coord[2]))
                y2 = str(int(coord[3]))
                score = str(float(coord[4]))
                w.write(x1 + ' ' + y1 + ' ' + x2 + ' ' + y2 + ' ' + score + "\n")
                w.close()
            if [int(coord[1]), int(coord[3]), int(coord[0]), int(coord[2])] == [0, 0, 0, 0]:
                print("未识别到物体")
            else:
                # cut = im[int(coord[1]):int(coord[3]), int(coord[0]):int(coord[2])]  # 裁切坐标为（y0:y1，x0:x1）
                cut = im[int(coord[1])-14:int(coord[3])+14, int(coord[0])-14:int(coord[2])+14]  # 裁切坐标为（y0:y1，x0:x1）
                score = coord[-1]
                save_name = image_name.split('.')[0] + '_' + str(j) + image_name[-4:]
                savecut(save_path, save_name, cut, score)