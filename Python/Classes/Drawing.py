import cv2
import mediapipe as mp
import numpy as np
from Classes.Index import Index
import math
import matplotlib.pyplot as plt
import os
from Classes.Pose import BiomechanicsImageInformation, PoseFeatures
import random

class Drawing:

    # BGR color
    color_black = (0, 0, 0)
    color_blue = (255, 0, 0)
    color_green = (0, 255, 0)
    color_red = (0, 0, 255)
    color_aqua = (255, 255, 0)
    color_magenta = (255, 0, 255)
    color_yellow = (0, 255, 255)
    color_white = (255, 255, 255)

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Face
    closed_polylines_parameters_i = dict(isClosed=True, color=color_green, thickness=1, lineType=cv2.LINE_AA)
    open_polylines_parameters_i = dict(isClosed=False, color=color_red, thickness=1, lineType=cv2.LINE_AA)
    iris_polylines_parameters_i = dict(isClosed=True, color=color_blue, thickness=2, lineType=cv2.LINE_AA)

    face_landmarks_parameters = dict(color=(0, 0, 255), thickness=1, circle_radius=1)
    face_connection_parameters = dict(color=(0, 255, 0), thickness=2)

    text_fps_parameters = dict(fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=color_red, fontScale=1, thickness=2, org=(50, 50), lineType=cv2.LINE_AA)
    text_parameters = dict(fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=color_blue, fontScale=1, thickness=2, org=(50, 200), lineType=cv2.LINE_AA)

    # Border
    border_parameters = dict(value=color_black, borderType=cv2.BORDER_CONSTANT)

    # Pose
    pose_landmarks_parameters = dict(color=(0, 0, 255), thickness=5, circle_radius=5)
    pose_connection_parameters = dict(color=(0, 255, 0), thickness=5)

    # Hand
    hand_landmarks_parameters = dict(color=(255, 0, 0), thickness=2, circle_radius=2)
    hand_connection_parameters = dict(color=(255, 255, 0), thickness=2)

class DrawingUtilities:

    @staticmethod
    def set_colors_tracking(tracking_max):

        colors = []

        for i in range(tracking_max):

            b = random.randint(0, 255)
            g = random.randint(0, 255)
            r = random.randint(0, 255)

            colors.append((b, g, r))

        return colors

    @staticmethod
    def put_highlight_rectangle(image, pt1, pt2, color, thickness):

        img = cv2.rectangle(img=image,
                            pt1=pt1,
                            pt2=pt2,
                            color=(0, 0, 0),
                            thickness=thickness + 3)

        img = cv2.rectangle(img=img,
                            pt1=pt1,
                            pt2=pt2,
                            color=color,
                            thickness=thickness)

        return img

    @staticmethod
    def put_highlight_text(image, text, org, scale_text, color, thickness):

        image = cv2.putText(image,
                            text=text,
                            org=org,
                            fontFace=cv2.FONT_HERSHEY_PLAIN,
                            fontScale=scale_text,
                            color=(0, 0, 0),
                            lineType=cv2.LINE_AA,
                            thickness=thickness + 3)

        image = cv2.putText(image,
                            text=text,
                            org=org,
                            fontFace=cv2.FONT_HERSHEY_PLAIN,
                            fontScale=scale_text,
                            color=color,
                            lineType=cv2.LINE_AA,
                            thickness=thickness)

        return image

    @staticmethod
    def draw_eyelid(image, results_i):

        # Draw lines
        mesh_points_left_eye_i = results_i[Index.index_eye_edge_loop_left_1]
        mesh_points_right_eye_i = results_i[Index.index_eye_edge_loop_right_1]

        cv2.polylines(img=image, pts=[mesh_points_left_eye_i], **Drawing.closed_polylines_parameters_i)
        cv2.polylines(img=image, pts=[mesh_points_right_eye_i], **Drawing.closed_polylines_parameters_i)

    @staticmethod
    def draw_iris(image, results_i):

        mesh_points_left_iris_i = results_i[Index.index_left_iris]
        mesh_points_right_iris_i = results_i[Index.index_right_iris]

        cv2.polylines(img=image, pts=[mesh_points_left_iris_i], **Drawing.iris_polylines_parameters_i)
        cv2.polylines(img=image, pts=[mesh_points_right_iris_i], **Drawing.iris_polylines_parameters_i)

    @staticmethod
    def draw_pose(image, results_mp, mp_holistic):

        Drawing.mp_drawing.draw_landmarks(image=image, landmark_list=results_mp.pose_landmarks, connections=mp_holistic.POSE_CONNECTIONS,
                                  landmark_drawing_spec=Drawing.mp_drawing.DrawingSpec(**Drawing.pose_landmarks_parameters),
                                  connection_drawing_spec=Drawing.mp_drawing.DrawingSpec(**Drawing.pose_connection_parameters))

    @staticmethod
    def draw_pose_pl(image, results_mp_pl, mp_holistic):

        Drawing.mp_drawing.draw_landmarks(image=image, landmark_list=results_mp_pl, connections=mp_holistic.POSE_CONNECTIONS,
                                  landmark_drawing_spec=Drawing.mp_drawing.DrawingSpec(**Drawing.pose_landmarks_parameters),
                                  connection_drawing_spec=Drawing.mp_drawing.DrawingSpec(**Drawing.pose_connection_parameters))

    @staticmethod
    def draw_face(image, results_mp, mp_face_mesh):

        if results_mp.multi_face_landmarks:
            for face_landmarks in results_mp.multi_face_landmarks:
                Drawing.mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks, connections=mp_face_mesh.FACEMESH_CONTOURS,
                                               landmark_drawing_spec=Drawing.mp_drawing.DrawingSpec(**Drawing.face_landmarks_parameters),
                                               connection_drawing_spec=Drawing.mp_drawing.DrawingSpec(**Drawing.face_connection_parameters))

    @staticmethod
    def draw_hand(image, results_mp, mp_holistic):

        Drawing.mp_drawing.draw_landmarks(image=image, landmark_list=results_mp.right_hand_landmarks, connections=mp_holistic.HAND_CONNECTIONS,
                                  landmark_drawing_spec=Drawing.mp_drawing.DrawingSpec(**Drawing.hand_landmarks_parameters),
                                  connection_drawing_spec=Drawing.mp_drawing.DrawingSpec(**Drawing.hand_connection_parameters))

        Drawing.mp_drawing.draw_landmarks(image=image, landmark_list=results_mp.left_hand_landmarks, connections=mp_holistic.HAND_CONNECTIONS,
                                  landmark_drawing_spec=Drawing.mp_drawing.DrawingSpec(**Drawing.hand_landmarks_parameters),
                                  connection_drawing_spec=Drawing.mp_drawing.DrawingSpec(**Drawing.hand_connection_parameters))

    @staticmethod
    def draw_biomechanics_informations(image, results_biomechanics_frame: dict, results_i_frame: dict):

        image_information = BiomechanicsImageInformation(results_i=results_i_frame)

        for key in image_information.dict.keys():

            if (np.isnan(results_biomechanics_frame[key])):

                value = "nan"

            else:

                value = round(results_biomechanics_frame[key], 1)

            if "r" in key[-1]:

                x = image_information.dict[key][0] - 90

            else:

                x = image_information.dict[key][0]

            cv2.putText(img=image,
                        text=str(value) + " [deg]",
                        org=(int(x), int(image_information.dict[key][1])),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=Drawing.color_red,
                        thickness=1,
                        lineType=cv2.LINE_AA)

    @staticmethod
    def draw_reference_system(frame, center, reference_system, scale, image_width, image_height, bio_info_dict: dict = None):

        center[0] *= image_width
        center[1] *= image_height

        cx = int(center[0])
        cy = int(center[1])
        cv2.circle(frame, (cx, cy), 3, (255, 255, 255), 2)

        reference_system *= scale
        xaxis = reference_system[:, 0]
        yaxis = reference_system[:, 1]
        zaxis = reference_system[:, 2]
        zaxis1 = reference_system[:, 2] * -1

        xp2 = xaxis[0] + cx
        yp2 = xaxis[1] + cy
        p2 = (int(xp2), int(yp2))
        cv2.line(frame, (cx, cy), p2, (0, 0, 255), 2)

        if bio_info_dict is not None:

            cv2.putText(img=frame,
                        text="x: " + str(round(bio_info_dict["trunk_roll"], 1)),
                        org=p2,
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(0, 0, 0),
                        thickness=2,
                        lineType=cv2.LINE_AA)

            cv2.putText(img=frame,
                        text="x: " + str(round(bio_info_dict["trunk_roll"], 1)),
                        org=p2,
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(0, 0, 255),
                        thickness=1,
                        lineType=cv2.LINE_AA)

        xp2 = yaxis[0] + cx
        yp2 = yaxis[1] + cy
        p2 = (int(xp2), int(yp2))
        cv2.line(frame, (cx, cy), p2, (0, 255, 0), 2)

        if bio_info_dict is not None:

            cv2.putText(img=frame,
                        text="y: " + str(round(bio_info_dict["trunk_pitch"], 1)),
                        org=p2,
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(0, 0, 0),
                        thickness=2,
                        lineType=cv2.LINE_AA)

            cv2.putText(img=frame,
                        text="y: " + str(round(bio_info_dict["trunk_pitch"], 1)),
                        org=p2,
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(0, 255, 0),
                        thickness=1,
                        lineType=cv2.LINE_AA)

        xp1 = zaxis1[0] + cx
        yp1 = zaxis1[1] + cy
        p1 = (int(xp1), int(yp1))
        xp2 = zaxis[0] + cx
        yp2 = zaxis[1] + cy
        p2 = (int(xp2), int(yp2))

        cv2.line(frame, p1, p2, (255, 0, 0), 2)
        cv2.circle(frame, p2, 3, (255, 0, 0), 2)

        if bio_info_dict is not None:

            cv2.putText(img=frame,
                        text="z: " + str(round(bio_info_dict["trunk_yaw"], 1)),
                        org=p2,
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(0, 0, 0),
                        thickness=2,
                        lineType=cv2.LINE_AA)

            cv2.putText(img=frame,
                        text="z: " + str(round(bio_info_dict["trunk_yaw"], 1)),
                        org=p2,
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(255, 0, 0),
                        thickness=1,
                        lineType=cv2.LINE_AA)

        return frame

    @staticmethod
    def plot_3D_pose_landmark_with_trunk(results_mp, mp_holistic, angle:int=50):

        landmarks = results_mp.pose_world_landmarks.landmark
        n_landmarks = len(landmarks)

        results_w = np.array([[-landmarks[k].z, landmarks[k].x, -landmarks[k].y] for k in range(n_landmarks)])

        O, trunk_reference_system = PoseFeatures.compute_trunk_reference_system(left_shoulder=results_w[11],
                                                                                right_shoulder=results_w[12],
                                                                                left_hip=results_w[23],
                                                                                right_hip=results_w[24])
        Drawing.mp_drawing.plot_landmarks(landmark_list=results_mp.pose_world_landmarks,
                                  connections=mp_holistic.POSE_CONNECTIONS,
                                  landmark_drawing_spec=Drawing.mp_drawing.DrawingSpec(**Drawing.pose_landmarks_parameters),
                                  connection_drawing_spec=Drawing.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2))

        xaxis = trunk_reference_system[:, 0].transpose()
        yaxis = trunk_reference_system[:, 1].transpose()
        zaxis = trunk_reference_system[:, 2].transpose()

        # fig = plt.figure(figsize=(8, 8))
        plt.tight_layout()
        fig = plt.gcf()
        ax = plt.gca()

        ax.scatter(O[0], O[1], O[2], color="black")

        ax.quiver(O[0], O[1], O[2], xaxis[0], xaxis[1], xaxis[2], color="red", length=0.2)
        ax.quiver(O[0], O[1], O[2], yaxis[0], yaxis[1], yaxis[2], color="green", length=0.2)
        ax.quiver(O[0], O[1], O[2], zaxis[0], zaxis[1], zaxis[2], color="blue", length=0.2)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim((-1, 1))
        ax.set_ylim((-1, 1))
        ax.set_zlim((-1, 1))

        ax.view_init(30, angle)

        return fig, ax

    @staticmethod
    def plot_3D_pose_landmark(results_mp, mp_holistic, angle:int = 50):

        Drawing.mp_drawing.plot_landmarks(landmark_list=results_mp.pose_world_landmarks, connections=mp_holistic.POSE_CONNECTIONS,
                                          landmark_drawing_spec=Drawing.mp_drawing.DrawingSpec(**Drawing.pose_landmarks_parameters),
                                          connection_drawing_spec=Drawing.mp_drawing.DrawingSpec(**Drawing.pose_landmarks_parameters))

        fig = plt.gcf()
        ax = plt.gca()
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim((-1, 1))
        ax.set_ylim((-1, 1))
        ax.set_zlim((-1, 1))

        ax.view_init(30, angle)
        plt.tight_layout()

        return fig, ax

    @staticmethod
    def plot_3D_face_landmark(results_mp, mp_face_mesh, angle: int = 50):

        Drawing.mp_drawing.plot_landmarks(landmark_list=results_mp.multi_face_landmarks[0], connections=mp_face_mesh.FACEMESH_CONTOURS,
                                          landmark_drawing_spec=Drawing.mp_drawing.DrawingSpec(**Drawing.face_landmarks_parameters),
                                          connection_drawing_spec=Drawing.mp_drawing.DrawingSpec(**Drawing.face_connection_parameters))

        fig = plt.gcf()
        ax = plt.gca()
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        ax.view_init(30, angle)
        plt.tight_layout()

        return fig, ax

    @staticmethod
    def draw_persons_box(image, persons, color):

        MARGIN = 20

        for j, person in enumerate(persons):

            start_point = (person[0] - MARGIN, person[1] - MARGIN)
            end_point = (person[2] + MARGIN, person[3] + MARGIN)
            image = cv2.rectangle(image, start_point, end_point, color=color, thickness=2)

    @staticmethod
    def draw_box_with_ids(image, persons, ids_results, colors):

        SCALE_ID = 0.3
        MARGIN = 20

        for i, person in enumerate(persons):
            if (ids_results[i] != -1):
                color = colors[int(ids_results[i])]
                start_point = (person[0] - MARGIN, person[1] - MARGIN)
                end_point = (person[2] + MARGIN, person[3] + MARGIN)
                image = cv2.rectangle(image, start_point, end_point, color=color, thickness=2)

                image = cv2.putText(image, str(ids_results[i]), (person[0], person[1]), cv2.FONT_HERSHEY_PLAIN, int(20 * SCALE_ID),
                                    color, int(20 * SCALE_ID), cv2.LINE_AA)

    @staticmethod
    def draw_box_with_confidence(image, persons, confs, color, text_pos: int = 2):

        MARGIN = 10
        SCALE_TEXT = 2
        THICKNESS_TEXT = 1

        for j, person in enumerate(persons):

            start_point = (person[0] - MARGIN, person[1] - MARGIN)
            end_point = (person[2] + MARGIN, person[3] + MARGIN)

            image = DrawingUtilities.put_highlight_text(image=image,
                                                        text="conf: {:.2f}".format(confs[j]),
                                                        org=(person[0] - MARGIN, person[1] - MARGIN * text_pos),
                                                        scale_text=SCALE_TEXT,
                                                        color=color,
                                                        thickness=THICKNESS_TEXT)

            image = DrawingUtilities.put_highlight_rectangle(image=image,
                                                             pt1=start_point,
                                                             pt2=end_point,
                                                             color=color,
                                                             thickness=THICKNESS_TEXT)

        return image

    @staticmethod
    def draw_box_with_emotion(image, persons, emotions, probs, color, text_pos: int = 2):

        MARGIN = 10
        SCALE_TEXT = 2
        THICKNESS_TEXT = 1

        for j, person in enumerate(persons):

            start_point = (person[0] - MARGIN, person[1] - MARGIN)
            end_point = (person[2] + MARGIN, person[3] + MARGIN)

            image = DrawingUtilities.put_highlight_text(image=image,
                                                        text="{} - {:.1f}".format(emotions[j], probs[j]),
                                                        org=(person[0] - MARGIN, person[1] - MARGIN * text_pos),
                                                        scale_text=SCALE_TEXT,
                                                        color=color,
                                                        thickness=THICKNESS_TEXT)

            image = DrawingUtilities.put_highlight_rectangle(image=image,
                                                             pt1=start_point,
                                                             pt2=end_point,
                                                             color=color,
                                                             thickness=THICKNESS_TEXT)

        return image



