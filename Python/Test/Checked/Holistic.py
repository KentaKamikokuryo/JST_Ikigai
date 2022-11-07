from Classes.PathInfo import PathInfo
from Classes.ProcessWebcam import *
import cv2
from ClassesML.Recognizer import Gaze

path_info = PathInfo()

process_pose = ProcessHolisticFrame()
gaze = Gaze()
fps = FPS()

which = ["pose",  "right_hand", "left_hand"]

cap = cv2.VideoCapture(0, apiPreference=cv2.CAP_ANY, params=[cv2.CAP_PROP_FRAME_WIDTH, 1280, cv2.CAP_PROP_FRAME_HEIGHT, 720])

with process_pose.pose as pose:

    while cap.isOpened():

        ret, image = cap.read()
        results_i, results_w, results_mp = process_pose.process_pose(image=image)

        fps.detect()
        fps.draw(image=image)

        pressedKey = cv2.waitKey(1) & 0xFF

        if pressedKey == ord('q'):
            break

        # Draw body
        if results_i is not None:

            DrawingUtilities.draw_pose(image=image, results_mp=results_mp, mp_holistic=process_pose.mp_holistic)
            DrawingUtilities.draw_hand(image=image, results_mp=results_mp, mp_holistic=process_pose.mp_holistic)

        cv2.imshow("Pose", image)
