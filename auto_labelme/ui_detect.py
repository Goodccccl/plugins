import os
import sys

import cv2 as cv
import numpy as np
from PIL import Image
import json
import base64
from typing import Tuple

from segment_anything import sam_model_registry, SamPredictor

from PyQt5.QtWidgets import QApplication, QMessageBox, QLabel, QWidget, QVBoxLayout, QPushButton, QFileDialog, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QRect, QRectF
import PyQt5.QtCore as QtCore

detected_objects = []
global_img = None

def create_labelme_annotations(detected_objectss, filename, img_height, img_width):
    """
    Function to create Labelme-style annotations from a list of detected objects
    :param detected_objects: List of tuples with format (top_left, bottom_right)
    :param filename: Name of the file
    :param img_height: Image height
    :param img_width: Image width
    :return: Dictionary in Labelme-style format
    """
    shapes = []
    for (top_left, bottom_right) in detected_objectss:
        shape = {
            "label": "default",
            "points": [list(top_left), list(bottom_right)],
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {},
        }
        shapes.append(shape)

    with open(filename, "rb") as f:
        img_data = f.read()
        img_data = base64.b64encode(img_data).decode('utf-8')

    labelme_annotations = {
        "version": "5.0.3",
        "flags": {},
        "shapes": shapes,
        "imagePath": filename,
        "imageData": img_data,
        "imageHeight": img_height,
        "imageWidth": img_width
    }
    
    detected_objects.clear()
    
    return labelme_annotations

def get_mask_predictor(sam_checkpoint, model_type, device):
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor

def mouse_callback(event, x, y, flags, params):
    global detected_objects
    image = global_img
    pts = params[2]

    predictor = params[0]
    win_name = params[1]
    history = params[3]
    if len(pts) == 0:
        if event == cv.EVENT_LBUTTONDOWN:
            history.append(image.copy())
            pts.append([x, y])
            cv.circle(image, (x, y), 3, (255, 255, 255), -1, cv.LINE_AA)
            cv.imshow(win_name, image)
            # print(pts)
    elif len(pts) == 1:
        if event == cv.EVENT_LBUTTONDOWN:
            pts.append([x, y])
            input_point = np.array(pts, dtype=np.int32)
            input_label = np.array([1, 1])
            masks, scores = mul_point(input_point, input_label, predictor)
            p1, p2 = generatr_box_from_mask(image, masks)
            detected_objects.append((p1, p2))
            # 使用p1, p2与已经存在的box进行对比, 如果面积相近的话就视为同一个而不进行增加
            # params[0] = apply_mask_on_image(image, masks)  # mask不是在原来的图像上进行操作的而是生成了新的图像
            cv.imshow(win_name, image)
            pts.clear()
        elif event == cv.EVENT_MOUSEMOVE:
            input_point = np.array(pts + [[x, y]], dtype=np.int32)
            input_label = np.array([1, 1])
            masks, scores = mul_point(input_point, input_label, predictor)
            image_copy = image.copy()
            p1, p2 = generatr_box_from_mask(image_copy, masks)
            image_copy = apply_mask_on_image(image_copy, masks)  # mask不是在原来的图像上进行操作的而是生成了新的图像
            cv.imshow(win_name, image_copy)

def mul_point(input_point, input_label, predictor):
    # 多个点 来确定某个物体
    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    masks, scores = process_mask(masks, scores)
    return masks, scores

def generatr_box_from_mask(image, mask, random_color=True):
    if random_color:
        color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
    else:
        color = (30, 144, 255)

    matrix = mask.squeeze().astype(np.uint8) * 255
    # 寻找轮廓
    contours, _ = cv.findContours(matrix, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    x_min = matrix.shape[1] - 1
    y_min = matrix.shape[0] - 1
    x_max = 0
    y_max = 0
    for cnt in contours:
        # 获取轮廓的外接矩形
        x, y, w, h = cv.boundingRect(cnt)
        # 更新最大外接矩形的坐标
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)
    cv.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2, cv.LINE_AA)
    return (x_min, y_min), (x_max, y_max)

def get_image(image_path):
    image = cv.imread(image_path)
    # return image
    return cv.cvtColor(image, cv.COLOR_BGR2RGB)

def process_mask(masks, scores):
    idx = np.argmax(scores, axis=-1)
    if len(masks.shape) == 4:
        # 多个批量
        return masks[np.arange(masks.shape[0]), idx], scores[np.arange(scores.shape[0]), idx]
    else:
        return masks[idx][None], scores[idx]
    
def apply_mask_on_image(image, mask, random_color=True):
    if random_color:
        color = np.concatenate([np.random.randint(0, 255, 3, dtype=np.uint8)], axis=0)
    else:
        color = np.array([30, 144, 255], dtype=np.uint8)
    h, w = mask.shape[-2:]
    mask_image = mask.reshape((h, w, 1)) * color.reshape((1, 1, -1))
    return cv.addWeighted(image, 1, mask_image, 0.65, 0)

def two_point_example(predictor, image_path, win_name="1"):
    global global_img, detected_objects
    image = get_image(image_path)
    height, width = image.shape[0], image.shape[1]
    predictor.set_image(image)
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    global_img = image
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    pts = []
    history = []
    cv.setMouseCallback(win_name, mouse_callback, param=[predictor, win_name, pts, history])
    cv.imshow(win_name, image)
    while True:
        if cv.getWindowProperty(win_name, cv.WND_PROP_VISIBLE) < 1:
            break
        key = cv.waitKey(0)
        if key == 9:  # 当用户按下"Tab"键时退出
            break
        elif key == 26:  # Ctrl+Z的ASCII码为26
            if len(history) > 0:
                image = history.pop()  # 从历史状态中取出最后一个状态，并在历史状态中删除它
                detected_objects.pop()
                global_img = image
                cv.imshow(win_name, image)
    cv.destroyAllWindows()
    cv.waitKey(0)
    cv.destroyAllWindows()
    return height, width

def get_predictor(checkpoint, model_type, device):
    predictor = get_mask_predictor(checkpoint, model_type, device)
    return predictor


# GUI代码部分

class MyWindow(QWidget):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.setWindowTitle('Label Window')
        self.predictor = get_predictor("sam_vit_l_0b3195.pth", "vit_l", "cuda")
        self.imagePath = None  # path to the image file
        self.imageList = []  # list of images in the folder
        self.imageIndex = -1  # current image index
        self.button_load = QPushButton('Load Image', self)
        self.button_next = QPushButton('Next Image', self)
        self.button_prev = QPushButton('Previous Image', self)
        self.button_exit = QPushButton('Exit', self)

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        layout.addWidget(self.button_load)
        layout.addWidget(self.button_next)
        layout.addWidget(self.button_prev)
        layout.addWidget(self.button_exit)
        self.setLayout(layout)

        self.button_load.clicked.connect(self.load_image)
        self.button_next.clicked.connect(self.next_image)
        self.button_prev.clicked.connect(self.prev_image)
        self.button_exit.clicked.connect(self.exit_application)
        
        # font = QFont('Times', 12, QFont.Bold)
        # self.setFont(font)

    @staticmethod
    def get_image_files(directory):
        extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif']
        return [directory + '/' + file for file in os.listdir(directory) if os.path.splitext(file)[1].lower() in extensions], \
            [file for file in os.listdir(directory) if os.path.splitext(file)[1].lower() in extensions]
        
    def _image_process(self):
        global detected_objects
        height, width = two_point_example(self.predictor, self.imagePath)
        labelme_annotations = create_labelme_annotations(detected_objects, self.imagePath, height, width)
        local = self.nameList[self.imageIndex].find('.')
        extend_name = self.nameList[self.imageIndex][local:]
        json_path = self.imagePath.replace(extend_name, ".json")
        with open(json_path, 'w') as f:
            json.dump(labelme_annotations, f, indent=4)
  
    def _cleanup(self):

        cv.destroyAllWindows()  # 关闭所有opencv窗口

    def _zoom_in(self):
        if self.imagePath is not None:
            img = cv.imread(self.imagePath)
            new_img = cv.resize(img, None, fx=1.25, fy=1.25, interpolation = cv.INTER_CUBIC)
            cv.imwrite(self.imagePath, new_img)
            self._image_process()

    def _zoom_out(self):
        if self.imagePath is not None:
            img = cv.imread(self.imagePath)
            new_img = cv.resize(img, None, fx=0.8, fy=0.8, interpolation = cv.INTER_AREA)
            cv.imwrite(self.imagePath, new_img)
            self._image_process()

    def exit_application(self):
        self._cleanup() 
        QApplication.instance().quit()  

    def _show_dialog_last(self):
        msgBox = QMessageBox()
        msgBox.setText("The last image!")
        msgBox.exec()
        
    def _show_dialog_first(self):
        msgBox = QMessageBox()
        msgBox.setText("The first image!")
        msgBox.exec()
        
    def load_image(self):
        image_path, _ = QFileDialog.getOpenFileName(self, "Open image", "", "Images (*.png *.jpg *.jpeg *.bmp *tif)")
        if image_path:
            self.imagePath = image_path
            folder = os.path.dirname(self.imagePath)
            self.imageList, self.nameList = self.get_image_files(folder)
            self.imageIndex = self.imageList.index(self.imagePath)
            print(self.imagePath)
            self._image_process()
            

    def next_image(self):
        if self.imageList and self.imageIndex != -1 and self.imageIndex + 1 < len(self.imageList):
            self.imageIndex = self.imageIndex + 1
            self.imagePath = self.imageList[self.imageIndex]
            print(self.imagePath)
            self._image_process()
        elif self.imageList and self.imageIndex + 1 >= len(self.imageList):
            self._show_dialog_last()
            

    def prev_image(self):
        if self.imageList and self.imageIndex != -1:
            self.imageIndex = (self.imageIndex - 1) % len(self.imageList)
            self.imagePath = self.imageList[self.imageIndex]
            print(self.imagePath)
            self._image_process()
        elif self.imageList and self.imageIndex - 1 < 0:
            self._show_dialog_first()



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyWindow()
    window.setGeometry(400, 400, 450, 420)
    window.show()
    app.exec_()
