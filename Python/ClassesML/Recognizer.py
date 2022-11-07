import matplotlib.pyplot as plt
import os, shutil, time
import json
import pandas as pd
from Classes.PathInfo import PathInfo
import cv2
import numpy as np
import mediapipe as mp
import rapidjson
from tabulate import tabulate
from sklearn import preprocessing
import math
import cv2
from abc import ABCMeta, abstractmethod
from Classes.Index import Index
from Classes.Drawing import Drawing

class Recognizer:

    def __init__(self):

        pass

    def set_results_w(self, results_w):

        self.results_w = results_w

    def set_results_i(self, results_i):

        self.results_i = results_i

class Gaze(Recognizer):

    def __init__(self):

        super().__init__()

    def detect(self):

        self.mesh_points_left_eye_i = self.results_i[Index.index_eye_edge_loop_left_1]
        self.mesh_points_right_eye_i = self.results_i[Index.index_eye_edge_loop_right_1]
        self.mesh_points_left_iris_i = self.results_i[Index.index_left_iris]
        self.mesh_points_right_iris_i = self.results_i[Index.index_right_iris]

        mesh_points_left_eye_center_i = np.mean(self.mesh_points_left_eye_i, axis=0)
        mesh_points_right_eye_center_i = np.mean(self.mesh_points_right_eye_i, axis=0)

        mesh_points_left_iris_center_i = np.mean(self.mesh_points_left_iris_i, axis=0)
        mesh_points_right_iris_center_i = np.mean(self.mesh_points_right_iris_i, axis=0)

        left_eye_width_i = np.max(self.mesh_points_left_eye_i[:, 0]) - np.min(self.mesh_points_left_eye_i[:, 0])

        look_right_detection_threshold = mesh_points_left_eye_center_i[0] - left_eye_width_i * 0.1
        look_left_detection_threshold = mesh_points_left_eye_center_i[0] + left_eye_width_i * 0.1

        if mesh_points_left_iris_center_i[0] < look_right_detection_threshold:
            self.label = "Look right"
        elif mesh_points_left_iris_center_i[0] > look_left_detection_threshold:
            self.label = "Look left"
        else:
            self.label = "Look center"

    def draw(self, image):

        image = cv2.putText(img=image, text=self.label, **Drawing.text_parameters)
