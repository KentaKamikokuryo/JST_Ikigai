import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, ion, show
from Classes.Drawing import *

class ProcessHolisticFrame():

    def __init__(self):

        self.drawing_parameters = Drawing()
        self.set_parameters()

    def set_parameters(self):

        self.mp_holistic = mp.solutions.holistic
        self.holistic_detection_parameters = dict(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1, min_tracking_confidence=0.2, smooth_landmarks=True)
        # self.pose = self.mp_holistic.Holistic(**self.holistic_detection_parameters)

    def process(self, image):

        with self.mp_holistic.Holistic(**self.holistic_detection_parameters) as pose:

            self.results_mp = pose.process(image)
            self.image_height, self.image_width, _ = image.shape

            self.n_landmark = len(self.results_mp.pose_landmarks.landmark)

            self.results_i = self._process_mp_to_image()
            self.results_w = self._process_mp_to_w()
            self.results_v = self._process_mp_to_visibility()

        return self.results_i, self.results_w, self.results_mp, self.results_v

    def _process_mp_to_image(self):

        results_i = np.array([[p.x * self.image_width, p.y * self.image_height] for p in self.results_mp.pose_landmarks.landmark]).astype(dtype=int)

        return results_i

    def _process_mp_to_w(self):

        landmarks = self.results_mp.pose_landmarks.landmark
        results_w = np.array([[-landmarks[k].z, landmarks[k].x, -landmarks[k].y] for k in range(self.n_landmark)])

        return results_w

    def _process_mp_to_visibility(self):

        landmarks = self.results_mp.pose_landmarks.landmark
        result_visibility = np.array([[landmarks[k].visibility] for k in range(self.n_landmark)])

        return result_visibility

class ProcessFaceFrame():

    def __init__(self):

        self.drawing_parameters = Drawing()
        self.set_parameters()

    def set_parameters(self):

        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_detection_parameters = dict(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.2, refine_landmarks=True, max_num_faces=1)

    def process(self, image):

        with self.mp_face_mesh.FaceMesh(**self.face_detection_parameters) as face_mesh:

            self.results_mp = face_mesh.process(image)
            self.image_height, self.image_width, _ = image.shape

            if self.results_mp.multi_face_landmarks:

                self.n_landmark = len(self.results_mp.multi_face_landmarks[0].landmark)

                self.results_i = self._process_mp_to_image()
                self.results_w = self._process_mp_to_w()
                self.results_v = self._process_mp_to_visibility()

            else:

                self.results_i = None
                self.results_w = None
                self.results_v = None

                print("Face not detected")

        return self.results_i, self.results_w, self.results_mp, self.results_v

    def _process_mp_to_image(self):

        results_i = np.array([[p.x * self.image_width, p.y * self.image_height] for p in self.results_mp.multi_face_landmarks[0].landmark]).astype(dtype=int)

        return results_i

    def _process_mp_to_w(self):

        landmarks = self.results_mp.multi_face_landmarks[0].landmark
        results_w = np.array([[-landmarks[k].z, landmarks[k].x, -landmarks[k].y] for k in range(self.n_landmark)])

        return results_w

    def _process_mp_to_visibility(self):

        landmarks = self.results_mp.multi_face_landmarks[0].landmark
        result_visibility = np.array([[landmarks[k].visibility] for k in range(self.n_landmark)])

        return result_visibility

class FPS:

    def __init__(self):

        self.prev_frame_time = 0
        self.new_frame_time = 0

    def detect(self):

        self.new_frame_time = time.time()
        self.fps = int(1 / (self.new_frame_time - self.prev_frame_time))
        self.prev_frame_time = self.new_frame_time

    def draw(self, image):

        cv2.putText(img=image, text=(str(self.fps)), **Drawing.text_fps_parameters)
