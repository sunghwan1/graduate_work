import os
import cv2
import time
import numpy as np
import tensorflow as tf


from queue import Queue
from threading import Thread
from multiprocessing import Process


from object_detection.utils import label_map_util as lmu
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import ops as utils_ops

#file import
import NumberPlate as NP



#define
time1 = time.time()
MIN_ratio = 0.85


MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
#MODEL_NAME = 'faster_rcnn_inception_v2_coco_2018_01_28'
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
    #frame = cv2.flip(frame, 0)
    #frame = cv2.flip(frame, 1)
    height, width, channel = frame.shape
    frame_expanded = np.expand_dims(frame, axis = 0)

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
    

    '''
    2 bicycle
    3 car
    4 motorcycle
    6 bus
    8 truck

    '''
    # objects = [] #리스트 생성
    for index, value in enumerate(classes[0]):
        object_dict = {}  # 딕셔너리
        if scores[0][index] > MIN_ratio:
            object_dict[(categories_index.get(value)).get('name').encode('utf8')] = \
                scores[0][index]
            # objects.append(object_dict) #리스트 추가

            '''visualize_boxes_and_labels_on_image_array box_size_info 이미지 정
            for box, color in box_to_color_map.items():
                ymin, xmin, ymax, xmax = box
            [index][0] [1]   [2]  [3]
            '''

            ymin = int((boxes[0][index][0] * height))
            xmin = int((boxes[0][index][1] * width))
            ymax = int((boxes[0][index][2] * height))
            xmax = int((boxes[0][index][3] * width))

            Result = frame[ymin:ymax, xmin:xmax]
            cv2.imwrite('car.jpg', Result)
            try:
                # print(NP.check())
                NP.number_recognition('car.jpg')
            except:
                print("응안돼")
            cv2.imshow('re', Result)
    # print(objects)

    cv2.imshow('cam', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break


cv2.destroyAllWindows()