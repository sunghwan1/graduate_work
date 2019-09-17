# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mygui.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!
import os
import cv2
import time
import numpy as np
import tensorflow as tf

from queue import Queue
from threading import Thread

from object_detection.utils import label_map_util as lmu
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import ops as utils_ops

#file import
import NumberPlate as NP

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.Qt import *


class Ui_Dialog(QWidget, object):

    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(874, 625)
        Dialog.setStatusTip("")
        Dialog.setSizeGripEnabled(False)
        Dialog.setModal(False)
        self.label_4 = QtWidgets.QLabel(Dialog)
        self.label_4.setGeometry(QtCore.QRect(20, 100, 401, 501))
        self.label_4.setStyleSheet("background-color: rgb(236, 187, 255);")
        self.label_4.setText("")
        self.label_4.setObjectName("label_4") #영상이 나올 화면
        self.frame = QtWidgets.QFrame(Dialog)
        self.frame.setGeometry(QtCore.QRect(440, 100, 411, 501))
        self.frame.setStyleSheet("background-color: rgb(170, 255, 255);")
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame") #결과화면
        self.textBrowser = QtWidgets.QTextBrowser(self.frame)
        self.textBrowser.setGeometry(QtCore.QRect(0, 10, 411, 481))
        self.textBrowser.setFrameShadow(QtWidgets.QFrame.Raised)
        self.textBrowser.setObjectName("textBrowser") #줄단위 결과창

        '''버튼 아이콘 생성자'''
        self.Cam_button = QtWidgets.QPushButton(Dialog)
        self.Cam_button.setGeometry(QtCore.QRect(20, 10, 71, 61))
        self.Cam_button.setStyleSheet("background-color: rgb(240, 240, 240);")
        self.Cam_button.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("../My_detection/mygui/image/camera.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.Cam_button.setIcon(icon)
        self.Cam_button.setIconSize(QtCore.QSize(50, 50))
        self.Cam_button.setObjectName("Cam_button")
        self.Cam_button.clicked.connect(self.Cam_button_clicked) #카메라 버튼이벤트 생성
        
        self.VIdeo_button = QtWidgets.QPushButton(Dialog)
        self.VIdeo_button.setGeometry(QtCore.QRect(100, 10, 71, 61))
        self.VIdeo_button.setStyleSheet("background-color: rgb(240, 240, 240);")
        self.VIdeo_button.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("../My_detection/mygui/image/play-button.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.VIdeo_button.setIcon(icon1)
        self.VIdeo_button.setIconSize(QtCore.QSize(50, 50))
        self.VIdeo_button.setObjectName("VIdeo_button")
        self.VIdeo_button.clicked.connect(self.VIdeo_button_clicked) #비디오 버튼이벤트 생성

        self.Tf_button = QtWidgets.QPushButton(Dialog)
        self.Tf_button.setGeometry(QtCore.QRect(180, 10, 71, 61))
        self.Tf_button.setStyleSheet("background-color: rgb(240, 240, 240);")
        self.Tf_button.setText("")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("../My_detection/mygui/image/tensorflow.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.Tf_button.setIcon(icon2)
        self.Tf_button.setIconSize(QtCore.QSize(50, 50))
        self.Tf_button.setObjectName("Tf_button")
        self.Tf_button.clicked.connect(self.Tf_button_clicked) #텐서플로 버튼이벤트 생성
        
        self.Res_button = QtWidgets.QPushButton(Dialog)
        self.Res_button.setGeometry(QtCore.QRect(260, 10, 71, 61))
        self.Res_button.setStyleSheet("background-color: rgb(240, 240, 240);")
        self.Res_button.setText("")
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap("../My_detection/mygui/image/result.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.Res_button.setIcon(icon3)
        self.Res_button.setIconSize(QtCore.QSize(50, 50))
        self.Res_button.setObjectName("Res_button")
        #self.Reass_button.clicked.connect(self.Res_button_clicked) #결과창 버튼이벤트 생성
        
        '''라벨생성자'''
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(40, 80, 41, 16))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(120, 80, 41, 16))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(190, 80, 51, 16))
        self.label_3.setObjectName("label_3")
        self.label_5 = QtWidgets.QLabel(Dialog)
        self.label_5.setGeometry(QtCore.QRect(280, 80, 41, 16))
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(Dialog)
        self.label_6.setGeometry(QtCore.QRect(500, 60, 81, 16))
        self.label_6.setObjectName("label_6")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "오일쇼크"))
        self.label.setText(_translate("Dialog", "카메라"))
        self.label_2.setText(_translate("Dialog", "동영상"))
        self.label_3.setText(_translate("Dialog", "텐서플로"))
        self.label_5.setText(_translate("Dialog", "결과창"))
        self.label_6.setText(_translate("Dialog", "인식한 번호판"))
##end UI set
        
    def Cam_button_clicked(self):
        th = Thread(self)
        th.changePixmap.connect(self.setImage)
        th.start()

        print('씨불탱1')

    def setImage(self, image):
        self.label_4.setPixmap(QtGui.QPixmap.fromImage(image))


    #end def Cam_button
        
    def VIdeo_button_clicked(self):

        print('씨불탱2')
    #end def Video_button
        
    def Tf_button_clicked(self):
        
        print('씨불탱3')
    #end def Tf_button
        
    def Res_button_clicked(self):
        
        print('씨불탱4')
    #end def Res_button



class Thread(QThread):
    changePixmap = pyqtSignal(QImage)

    def run(self):

        time1 = time.time()
        MIN_ratio = 0.8

        #MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
        MODEL_NAME = 'faster_rcnn_inception_v2_coco_2018_01_28'
        GRAPH_FILE_NAME = 'frozen_inference_graph.pb'
        LABEL_FILE = 'data/mscoco_label_map.pbtxt'
        NUM_CLASSES = 90
        #end define

        label_map = lmu.load_labelmap(LABEL_FILE)
        categories = lmu.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        categories_index = lmu.create_category_index(categories)

        print("call label_map & categories : %0.5f" % (time.time() - time1))

        graph_file = MODEL_NAME + '/' + GRAPH_FILE_NAME

        #thread function
        def find_detection_target(categories_index, classes, scores):
            time1_1 = time.time() #스레드함수 시작시간
            print("스레드 시작")

            objects = [] #리스트 생성
            for index, value in enumerate(classes[0]):
                object_dict = {} #딕셔너리
                if scores[0][index] > MIN_ratio:
                    object_dict[(categories_index.get(value)).get('name').encode('utf8')] = \
                            scores[0][index]
                    objects.append(object_dict) #리스트 추가
            print(objects)

            print("스레드 함수 처리시간 %0.5f" & (time.time() - time1_1))

        #end thread function

        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name = '')

            sses = tf.Session(graph = detection_graph)

        print("store in memoey time : %0.5f" % (time.time() - time1))

        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        print("make tensor time : %0.5f" % (time.time() - time1))


        #capture = cv2.VideoCapture(0)
        capture = cv2.VideoCapture("20190916_162900.mp4")
        prevtime = 0

        #thread_1 = Process(target = find_detection_target, args = (categories_index, classes, scores))#쓰레드 생성
        print("road Video time : %0.5f" % (time.time() - time1))

        while True:
            ret, frame = capture.read()
            frame_expanded = np.expand_dims(frame, axis = 0)
            height, width, channel = frame.shape

            #프레임 표시
            curtime = time.time()
            sec = curtime - prevtime
            prevtime = curtime
            fps = 1/ sec
            str = "FPS : %0.1f" % fps
            cv2.putText(frame, str, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
            #end 프레임

            (boxes, scores, classes, nums) = sses.run( #np.ndarray
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: frame_expanded}
                )#end sses.run()

            vis_util.visualize_boxes_and_labels_on_image_array(
                frame,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                categories_index,
                use_normalized_coordinates = True,
                min_score_thresh = MIN_ratio,#최소 인식률
                line_thickness = 2)#선두께

            if ret:
                # https://stackoverflow.com/a/55468544/6622587
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QtGui.QImage(rgbImage.data, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)
                p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)

            # objects = [] #리스트 생성
            for index, value in enumerate(classes[0]):
                object_dict = {}  # 딕셔너리
                if scores[0][index] > MIN_ratio:
                    object_dict[(categories_index.get(value)).get('name').encode('utf8')] = \
                        scores[0][index]
                    # objects.append(object_dict) #리스트 추가

                    #visualize_boxes_and_labels_on_image_array box_size_info 이미지 정
                    #for box, color in box_to_color_map.items():
                    #    ymin, xmin, ymax, xmax = box
                    #[index][0] [1]   [2]  [3]

                    ymin = int((boxes[0][index][0] * height))
                    xmin = int((boxes[0][index][1] * width))
                    ymax = int((boxes[0][index][2] * height))
                    xmax = int((boxes[0][index][3] * width))

                    Result = frame[ymin:ymax, xmin:xmax]
                    cv2.imwrite('car.jpg', Result)
                    try:
                        result_chars = NP.number_recognition('car.jpg')
                        ui.label_6.setText(result_chars)
                        # print(NP.check())

                    except:
                        print("응안돼")
                    #cv2.imshow('re', Result)
            # print(objects)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)

    changePixmap = pyqtSignal(QImage)

    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()

    sys.exit(app.exec_())

'''
from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(874, 625)
        Dialog.setStatusTip("")
        Dialog.setSizeGripEnabled(False)
        Dialog.setModal(False)
        
        self.label_4 = QtWidgets.QLabel(Dialog)
        self.label_4.setGeometry(QtCore.QRect(20, 100, 401, 501))
        self.label_4.setStyleSheet("background-color: rgb(236, 187, 255);")
        self.label_4.setText("")
        self.label_4.setObjectName("label_4") #영상이 나올 화면
        self.frame = QtWidgets.QFrame(Dialog)
        self.frame.setGeometry(QtCore.QRect(440, 100, 411, 501))
        self.frame.setStyleSheet("background-color: rgb(170, 255, 255);")
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame") #결과화면
        self.textBrowser = QtWidgets.QTextBrowser(self.frame)
        self.textBrowser.setGeometry(QtCore.QRect(0, 10, 411, 481))
        self.textBrowser.setFrameShadow(QtWidgets.QFrame.Raised)
        self.textBrowser.setObjectName("textBrowser") #줄단위 결과창

        #버튼 아이콘 생성자
        self.Cam_button = QtWidgets.QPushButton(Dialog)
        self.Cam_button.setGeometry(QtCore.QRect(20, 10, 71, 61))
        self.Cam_button.setStyleSheet("background-color: rgb(240, 240, 240);")
        self.Cam_button.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("../My_detection/mygui/image/camera.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.Cam_button.setIcon(icon)
        self.Cam_button.setIconSize(QtCore.QSize(50, 50))
        self.Cam_button.setObjectName("Cam_button")
        
        self.VIdeo_button = QtWidgets.QPushButton(Dialog)
        self.VIdeo_button.setGeometry(QtCore.QRect(100, 10, 71, 61))
        self.VIdeo_button.setStyleSheet("background-color: rgb(240, 240, 240);")
        self.VIdeo_button.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("../My_detection/mygui/image/play-button.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.VIdeo_button.setIcon(icon1)
        self.VIdeo_button.setIconSize(QtCore.QSize(50, 50))
        self.VIdeo_button.setObjectName("VIdeo_button")
        
        self.Tf_button = QtWidgets.QPushButton(Dialog)
        self.Tf_button.setGeometry(QtCore.QRect(180, 10, 71, 61))
        self.Tf_button.setStyleSheet("background-color: rgb(240, 240, 240);")
        self.Tf_button.setText("")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("../My_detection/mygui/image/tensorflow.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.Tf_button.setIcon(icon2)
        self.Tf_button.setIconSize(QtCore.QSize(50, 50))
        self.Tf_button.setObjectName("Tf_button")
        
        self.Res_button = QtWidgets.QPushButton(Dialog)
        self.Res_button.setGeometry(QtCore.QRect(260, 10, 71, 61))
        self.Res_button.setStyleSheet("background-color: rgb(240, 240, 240);")
        self.Res_button.setText("")
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap("../My_detection/mygui/image/result.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.Res_button.setIcon(icon3)
        self.Res_button.setIconSize(QtCore.QSize(50, 50))
        self.Res_button.setObjectName("Res_button")
        
        #라벨생성자
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(40, 80, 41, 16))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(120, 80, 41, 16))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(190, 80, 51, 16))
        self.label_3.setObjectName("label_3")
        self.label_5 = QtWidgets.QLabel(Dialog)
        self.label_5.setGeometry(QtCore.QRect(280, 80, 41, 16))
        self.label_5.setObjectName("label_5")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "오일쇼크"))
        self.label.setText(_translate("Dialog", "카메라"))
        self.label_2.setText(_translate("Dialog", "동영상"))
        self.label_3.setText(_translate("Dialog", "텐서플로"))
        self.label_5.setText(_translate("Dialog", "결과창"))
##end UI set
'''
