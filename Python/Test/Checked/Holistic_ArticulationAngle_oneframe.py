import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
import mediapipe as mp
import torch
from Classes.PathInfo import PathInfo
from Classes.Pose import BiomechanicsInformation, PresenceVisibility, PoseFeatures, Rotation, Euler
from Classes.Drawing import DrawingUtilities


"""
For test, the code gets one frame, and then compute with this information.
"""

mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_holistic = mp.solutions.holistic # Mediapipe Solutions

def _quiver(ax, origin, vector, color):
    ax.quiver(origin[0], origin[1], origin[2], vector[0], vector[1], vector[2], color=color, length=0.2)

holistic_detection_parameters = dict(static_image_mode=False,
                                     min_detection_confidence=0.3,
                                     model_complexity=1,
                                     min_tracking_confidence=0.2,
                                     smooth_landmarks=True)

# path_info = PathInfo()
#
# # Sport video pose test (Face not detected properly)
# video_path = path_info.path_data_test + "Video_test.mp4"
video_path = "C:\\Users\\Kenta Kamikokuryo\\Desktop\\IKIGAI\\Ikigai\\Python\\TestVideo\\Video_test.mp4"
cap = cv2.VideoCapture(video_path)

data_frame_dict = {}
print("Extracting frame from 1 frame")
cap.set(1, 0)
ret, frame = cap.read()

# Recolor Feed from RGB to BGR
image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# making image writeable to false improves prediction
image.flags.writeable = False

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
with mp_holistic.Holistic(**holistic_detection_parameters) as holistic:

    results = holistic.process(image)

    n_landmark = len(results.pose_world_landmarks.landmark)

    landmarks = results.pose_world_landmarks.landmark
    results_w = np.array([[-landmarks[k].z, landmarks[k].x, -landmarks[k].y] for k in range(n_landmark)])

    results_v = np.array([[landmarks[k].visibility] for k in range(n_landmark)])
    presenceVisibility = PresenceVisibility(results_v=results_v, threshold=0.8)

    poseArticulationAngle = BiomechanicsInformation(results_hollistic_w=results_w, presenceVisibility=presenceVisibility)
    dict = poseArticulationAngle.dict

    O, trunk_reference_system = PoseFeatures.compute_trunk_reference_system(left_shoulder=results_w[11],
                                                                            right_shoulder=results_w[12],
                                                                            left_hip=results_w[23],
                                                                            right_hip=results_w[24])

    # For confirming
    euler = Rotation.rotation_matrix_to_euler_angles(rotation_matrix=trunk_reference_system,
                                                     seq="ZYX",
                                                     to_degrees=True)

    # For confirming
    R = Euler.euler_angles_to_rotation_matrix(euler=euler, seq="ZYX", to_rad=True)


    # origin_w = PoseFeatures.get_origin_w()
    #
    # rotation_matrix = Rotation.compute_relative_rotation_matrix(origin_w,
    #                                                             trunk_reference_system)

    mp_drawing.plot_landmarks(landmark_list=results.pose_world_landmarks, connections=mp_holistic.POSE_CONNECTIONS,
                              landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1),
                              connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2))

    xaxis = trunk_reference_system[:, 0].transpose()
    yaxis = trunk_reference_system[:, 1].transpose()
    zaxis = trunk_reference_system[:, 2].transpose()

    fig = plt.gcf()
    ax = plt.gca()

    ax.scatter(O[0], O[1], O[2], color="black")

    _quiver(ax=ax,
            origin=O,
            vector=xaxis, color="red")
    _quiver(ax=ax,
            origin=O,
            vector=yaxis, color="green")
    _quiver(ax=ax,
            origin=O,
            vector=zaxis, color="blue")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim((-1, 1))
    ax.set_ylim((-1, 1))
    ax.set_zlim((-1, 1))

    angle = 50
    ax.view_init(30, angle)
    plt.tight_layout()

    plt.show()


