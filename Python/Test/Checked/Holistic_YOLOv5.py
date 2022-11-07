import os
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import mediapipe as mp
import torch
from Classes.PathInfo import PathInfo
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

# Model = YOLO version 5
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# since we are only intrested in detecting person
yolo_model.classes = [0]

mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_holistic = mp.solutions.holistic # Mediapipe Solutions

holistic_detection_parameters = dict(static_image_mode=False,
                                     min_detection_confidence=0.3,
                                     model_complexity=1,
                                     min_tracking_confidence=0.2,
                                     smooth_landmarks=True)

path_info = PathInfo()

# Sport video pose test (Face not detected properly)
video_path = path_info.path_data_test + "Video_test.mp4"
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Recolor Feed from RGB to BGR
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # making image writeable to false improves prediction
    image.flags.writeable = False

    # Output the result with YOLO for detecting persons
    result = yolo_model(image)
    temp = result.xyxy[0].tolist()  # For seeing the temporal result

    # Recolor image back to BGR for rendering
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # image = cv2.resize(image, (int(384), int(640)))
    # print(result.xyxy)  # img1 predictions (tensor)

    # This array will contain crops of images incase we need it
    img_list = []

    # we need some extra margin bounding box for human crops to be properly detected
    MARGIN = 10

    # Crop the part of the image that detects humans from the YOLO results.
    for (xmin, ymin, xmax, ymax, confidence, clas) in result.xyxy[0].tolist():

        with mp_holistic.Holistic(**holistic_detection_parameters) as holistic:

            img = image[int(ymin) + MARGIN:int(ymax) + MARGIN, int(xmin) + MARGIN:int(xmax) + MARGIN:]

            results = holistic.process(img)

            # 1. Draw face landmarks
            mp_drawing.draw_landmarks(image[int(ymin) + MARGIN:int(ymax) + MARGIN, int(xmin) + MARGIN:int(xmax) + MARGIN:],
                                      results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                      mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                      mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                      )

            # 2. Right hand
            mp_drawing.draw_landmarks(image[int(ymin) + MARGIN:int(ymax) + MARGIN, int(xmin) + MARGIN:int(xmax) + MARGIN:],
                                      results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                      )

            # 3. Left Hand
            mp_drawing.draw_landmarks(image[int(ymin) + MARGIN:int(ymax) + MARGIN, int(xmin) + MARGIN:int(xmax) + MARGIN:],
                                      results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                      )

            # 4. Pose Detections
            mp_drawing.draw_landmarks(image[int(ymin) + MARGIN:int(ymax) + MARGIN, int(xmin) + MARGIN:int(xmax) + MARGIN:],
                                      results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )

            cv2.imshow("", image)

            img_list.append(image[int(ymin):int(ymax), int(xmin):int(xmax):])

    ## Code to quit the video incase you are using the webcam
    cv2.imshow('Activity recognition', image)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()