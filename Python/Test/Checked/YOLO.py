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
yolo_model.classes = [0]

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
    crop = result.crop(save=False)
    print(crop)

    # Recolor image back to BGR for rendering
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # image = cv2.resize(image, (int(384), int(640)))
    # print(result.xyxy)  # img1 predictions (tensor)

    # This array will contain crops of images incase we need it
    img_list = []
    # print(result.xyxy[0].tolist())

    MARGIN = 10

    for (xmin, ymin, xmax, ymax, confidence, clas) in result.xyxy[0].tolist():
        # Media pose prediction ,we are

        end_point = (int(xmax) + MARGIN, int(ymax) + MARGIN)
        start_point = (int(xmin) + MARGIN, int(ymin) + MARGIN)

        image = cv2.rectangle(image, start_point, end_point, color=(255, 0, 0), thickness=2)


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
