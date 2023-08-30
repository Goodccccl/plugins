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
    score = res[
        res >= template_threshold]  # 大于模板阈值的目标置信度cv2.matchTemplate(img_gray, template_img, cv2.TM_CCOEFF_NORMED)[res >= template_threshold]完整写法
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


def savecut(name, file, cut):
    isExists = os.path.exists("./" + str(name) + '/')
    if not isExists:  # 判断如果文件不存在,则创建
        os.makedirs("./" + str(name) + '/')
        print("./" + str(name) + '/' + "目录创建成功")

    out = str(file.split('.')[0])
    filename = "./" + name + '/%s.jpg' % (out)
    print('已裁切图片：' + filename)
    cv2.imwrite(filename, cut)


def camera(camera, save_path):
    cap = cv2.VideoCapture(camera)  # 打开摄像头
    '''
    frame_width=cap.get(cv2.CAP_PROP_FRAME_WIDTH)#获得视频尺寸——宽
    frame_height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)#高
    fps = cap.get(cv2.CAP_PROP_FPS)#获取视频fps
    print("[INFO]视频FPS:{}".format(fps))
    print("[INFO]视频宽:{}".format(frame_width))
    print("[INFO]视频高:{}".format(frame_height))
    #frame_all_fps=cap.get(cv2.CAP_PROP_FRAME_COUNT)#获取视频总帧数
    #print("[INFO]视频总帧数:{}".format(frame_all_fps))
    '''
    if not cap.isOpened():
        print("未打开摄像头!")

    else:
        print("已打开摄像头，任务开始!")
        frame_index = 1  # 图片计数

        retval, frame = cap.read()  # 这里是读取视频帧，第一个参数输出是否识别到，第二个参数输出识别到的图像帧
        while retval:  # 当读到图像帧时
            if cv2.waitKey(1) & 0xFF == ord('q'):  # 中断采集
                break
            if frame_index == 101:  # 拍摄多少张停止
                break
            print(frame_index)
            cv2.imshow("camera_frame", frame)  # 显示视频帧
            cv2.waitKey(100)  # 间隔100ms拍摄一张图片

            save1 = str(save_path)  # jpg格式图片存放文件夹的路径
            isExists = os.path.exists(save1)
            if not isExists:  # 判断如果文件不存在,则创建
                os.makedirs(save1)
            frame_name1 = f"camera_frame_{frame_index}.jpg"
            frame_name2 = str(frame_name1.split('.')[0])
            save1 = save1 + frame_name1
            retval, frame = cap.read()
            cv2.imwrite(save1, frame)  # 保存拍摄的图片

            file1 = os.path.join(save1)
            img_rgb = cv2.imread(file1)  # 需要检测的图片
            img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)  # 转化成灰色
            template_img = cv2.imread('mb.jpg', 0)  # 模板小图
            template_threshold = 0.5  # 模板置信度
            dets = template(img_gray, template_img, template_threshold)
            for coord in dets:
                cv2.rectangle(img_rgb, (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3])), (0, 0, 255), 2)
                saveresult('save', frame_name2, img_rgb)
                if [int(coord[1]), int(coord[3]), int(coord[0]), int(coord[2])] == [0, 0, 0, 0]:
                    print("未识别到物体")
                else:
                    cut = img_rgb[int(coord[1]):int(coord[3]), int(coord[0]):int(coord[2])]  # 裁切坐标为（y0:y1，x0:x1）
                    savecut('savepicture', frame_name2, cut)

            frame_index += 1

    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    camera(0, "./images/")  # 启动相机然后保存拍摄的图片

    print("***********************************************************************************")
    print('**                              已完成模板匹配任务                                  **')
    print("***********************************************************************************")

"""
参数                      propld	功能
cv2.CAP_PROP_POS_MSEC	    0	视频文件的当前位置（以毫秒为单位）或视频捕获时间戳
cv2.CAP_PROP_POS_FRAMES   	1	基于0的索引将被解码/捕获下一帧
cv2.CAP_PROP_POS_AVI_RATIO	2	视频文件的相对位置：0 - 视频的开始，1 - 视频的结束
cv2.CAP_PROP_FRAME_WIDTH	3	帧的宽度
cv2.CAP_PROP_FRAME_HEIGHT	4	帧的高度
cv2.CAP_PROP_FPS	        5	帧速
cv2.CAP_PROP_FOURCC	        6	4个字符表示的视频编码器格式
cv2.CAP_PROP_FRAME_COUNT	7	帧数
cv2.CAP_PROP_FORMAT	        8	byretrieve()返回的Mat对象的格式
cv2.CAP_PROP_MODE	        9	指示当前捕获模式的后端特定值
cv2.CAP_PROP_BRIGHTNESS	    10	图像的亮度（仅适用于相机）
cv2.CAP_PROP_CONTRAST	    11	图像对比度（仅适用于相机）
cv2.CAP_PROP_SATURATION	    12	图像的饱和度（仅适用于相机）
cv2.CAP_PROP_HUE	        13	图像的色相（仅适用于相机）
cv2.CAP_PROP_GAIN	        14	图像的增益（仅适用于相机）
cv2.CAP_PROP_EXPOSURE	    15	曝光（仅适用于相机）
cv2.CAP_PROP_CONVERT_RGB	16	表示图像是否应转换为RGB的布尔标志
cv2.CAP_PROP_WHITE_BALANCE	17	目前不支持
cv2.CAP_PROP_RECTIFICATION	18	立体摄像机的整流标志
"""
