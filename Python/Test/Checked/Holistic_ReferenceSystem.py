import mediapipe as mp
import cv2
import numpy as np
from Python.Classes.Pose import Euler, Rotation, PoseFeatures
from Python.Classes.Drawing import DrawingUtilities

mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_holistic = mp.solutions.holistic # Mediapipe Solutions
holistic_detection_parameters = dict(static_image_mode=False,
                                     min_detection_confidence=0.5,
                                     model_complexity=1,
                                     min_tracking_confidence=0.2,
                                     smooth_landmarks=True)

pose = mp_holistic.Holistic(**holistic_detection_parameters)

scale = 50
focal_length = 950

cap = cv2.VideoCapture(0)

def check_presence_trunk(array_visibility: np.array, threshold: float = 0.8) -> bool:

    is_reference_system = False

    shoulder_visibility_l = array_visibility[11]
    shoulder_visibility_r = array_visibility[12]
    hip_visibility_l = array_visibility[23]
    hip_visibility_r = array_visibility[24]

    if (shoulder_visibility_l > threshold and shoulder_visibility_r > threshold and hip_visibility_l > threshold and hip_visibility_r > threshold):

        is_reference_system = True

    return is_reference_system

def check_presence_head(array_visibility: np.array, threshold: float = 0.7) -> bool:

    is_reference_system = False

    nose_visibility = array_visibility[0]
    eye_inner_visibility_l = array_visibility[1]
    eye_inner_visibility_r = array_visibility[4]
    ear_visibility_l = array_visibility[7]
    ear_visibility_r = array_visibility[8]
    mouth_visibility_l = array_visibility[9]
    mouth_visibility_r = array_visibility[10]

    if (nose_visibility > threshold and eye_inner_visibility_l > threshold and eye_inner_visibility_r > threshold and ear_visibility_l > threshold and ear_visibility_r > threshold and mouth_visibility_l > threshold and mouth_visibility_r > threshold):

        is_reference_system = True

    return is_reference_system


# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

    while cap.isOpened():
        ret, frame = cap.read()

        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make Detections
        results_mp = pose.process(image)
        image_height, image_width, _ = image.shape

        # print(results.face_landmarks)

        # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks

        # Recolor image back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # # 1. Draw face landmarks
        # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
        #                           mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
        #                           mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
        #                           )

        # # 2. Right hand
        # mp_drawing.draw_landmarks(image, results_mp.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        #                           mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
        #                           mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
        #                           )
        #
        # # 3. Left Hand
        # mp_drawing.draw_landmarks(image, results_mp.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        #                           mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
        #                           mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
        #                           )

        # 4. Pose Detections
        mp_drawing.draw_landmarks(image, results_mp.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                  )

        # for plotting landmarks
        # x and y: Landmark coordinates normalized to [0.0, 1.0] by the image width and height respectively.
        # z: Should be discarded as currently the model is not fully trained to predict depth, but this is something on the roadmap.
        landmarks = results_mp.pose_landmarks.landmark
        n_landmark = len(landmarks)

        # x, y and z: Real-world 3D coordinates in meters with the origin at the center between hips.
        # visibility: Identical to that defined in the corresponding pose_landmarks.
        # landmarks = results_mp.pose_w_landmarks.landmark
        # n_landmark = len(landmarks)

        results_pose = np.array([[landmarks[k].x, landmarks[k].y, landmarks[k].z] for k in range(n_landmark)])
        print("result_pose: " + str(results_pose))

        result_visibility = np.array([[landmarks[k].visibility] for k in range(n_landmark)])
        print("result_visibility: " + str(result_visibility))

        # result_visibility_w = np.array([[landmarks_w[k].visibility] for k in range(n_landmark_w)])
        # print("result_visibility: " + str(result_visibility_w))


        is_reference_system = check_presence_trunk(array_visibility=result_visibility, threshold=0.75)
        print("If it is able to plot reference system of trunk or not: " + str(is_reference_system))

        if (is_reference_system):

            left_shoulder = results_pose[11]
            right_shoulder = results_pose[12]
            left_hip = results_pose[23]
            right_hip = results_pose[24]

            O_image_trunk, trunk_reference_system = PoseFeatures.compute_trunk_reference_system(left_shoulder=left_shoulder,
                                                                                                right_shoulder=right_shoulder,
                                                                                                left_hip=left_hip,
                                                                                                right_hip=right_hip)

            image = DrawingUtilities.draw_reference_system(frame=image,
                                                           center=O_image_trunk,
                                                           reference_system=trunk_reference_system,
                                                           scale=scale,
                                                           image_width=image_width,
                                                           image_height=image_height)

        is_reference_system_head = check_presence_head(array_visibility=result_visibility, threshold=0.75)
        print("If it is able to plot reference system of head or not: " + str(is_reference_system_head))

        if (is_reference_system_head):

            nose = results_pose[0]
            left_eye_inner = results_pose[1]
            right_eye_inner = results_pose[4]
            left_ear = results_pose[7]
            right_ear = results_pose[8]
            mouth_left = results_pose[9]
            mouth_right = results_pose[10]

            O_image_head, head_reference_system = PoseFeatures.compute_head_reference_system(nose=nose,
                                                                                             left_eye_inner=left_eye_inner,
                                                                                             right_eye_inner=right_eye_inner,
                                                                                             left_ear=left_ear,
                                                                                             right_ear=right_ear,
                                                                                             mouth_left=mouth_left,
                                                                                             mouth_right=mouth_right)

            image = DrawingUtilities.draw_reference_system(frame=image,
                                                           center=O_image_head,
                                                           reference_system=head_reference_system,
                                                           scale=scale,
                                                           image_width=image_width,
                                                           image_height=image_height)

        cv2.imshow('Raw Webcam Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()