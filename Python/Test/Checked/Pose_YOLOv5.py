import os
import matplotlib.pyplot as plt

from PIL import Image
import cv2

import mediapipe as mp
from mediapipe.python.solutions import pose as mp_pose

# PyTorch Hub
import torch

# Model

names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
        'teddy bear', 'hair drier', 'toothbrush']

yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# since we are only intrested in detecting person
yolo_model.classes = [0]

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose_detection_parameter = dict(static_image_mode=False,
                                min_detection_confidence=0.3,
                                model_complexity=1,
                                min_tracking_confidence=0.2,
                                smooth_landmarks=True)

# # get the dimension of the video
# cap = cv2.VideoCapture("C:\\Users\\Kenta Kamikokuryo\\Desktop\\IKIGAI\\Movie\\Demo4.LRV")
# while cap.isOpened():
#     ret, frame = cap.read()
#     h, w, _ = frame.shape
#     size = (w, h)
#     print(size)
#     break

# for webacam cv2.VideoCapture(NUM) NUM -> 0,1,2 for primary and secondary webcams..

# For saving the video file as output.avi
# out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 20, size)
cap = cv2.VideoCapture(0, apiPreference=cv2.CAP_ANY, params=[cv2.CAP_PROP_FRAME_WIDTH, 1280, cv2.CAP_PROP_FRAME_HEIGHT, 720])

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    # Recolor Feed from RGB to BGR
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # making image writeable to false improves prediction
    image.flags.writeable = False

    result = yolo_model(image)

    # Recolor image back to BGR for rendering
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # image = cv2.resize(image, (int(384), int(640)))
    # print(result.xyxy)  # img1 predictions (tensor)

    # This array will contain crops of images incase we need it
    img_list = []

    # we need some extra margin bounding box for human crops to be properly detected
    MARGIN = 10

    for (xmin, ymin, xmax, ymax, confidence, clas) in result.xyxy[0].tolist():
        with mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.3) as pose:
            # Media pose prediction ,we are
            results = pose.process(image[int(ymin) + MARGIN:int(ymax) + MARGIN, int(xmin) + MARGIN:int(xmax) + MARGIN:])

            # Draw landmarks on image, if this thing is confusing please consider going through numpy array slicing
            mp_drawing.draw_landmarks(
                image[int(ymin) + MARGIN:int(ymax) + MARGIN, int(xmin) + MARGIN:int(xmax) + MARGIN:],
                results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )
            img_list.append(image[int(ymin):int(ymax), int(xmin):int(xmax):])
    # cv2_imshow(image)

    # writing in the video file
    # out.write(image)

    ## Code to quit the video incase you are using the webcam
    cv2.imshow('Activity recognition', image)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
# out.release()
cv2.destroyAllWindows()
