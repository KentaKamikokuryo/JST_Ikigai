import mediapipe as mp
import numpy as np
from abc import ABC, abstractmethod


class FaceMeshFrame():

    def __init__(self):

        self._mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh_parameters = {"max_num_faces": 1,
                                     "refine_landmarks": True,
                                     "min_detection_confidence": 0.5,
                                     "min_tracking_confidence": 0.5}

    def infer(self, image, person_face_results, ids_results):

        MARGIN = 10
        self._set_results_dict()

        for i, face in enumerate(person_face_results):

            id = ids_results[i]

            if (id != -1):

                xmin = face[0]
                xmax = face[2]
                ymin = face[1]
                ymax = face[3]

                img = image[(int(ymin) - MARGIN):(int(ymax) + MARGIN), (int(xmin) - MARGIN):(int(xmax) + MARGIN)]

                with self._mp_face_mesh.FaceMesh(**self.face_mesh_parameters) as face_mesh:

                    self._results_mp[id] = face_mesh.process(img)

                    self._process(id=id, image=image)

        return self._results_mp

    # region To process the mp results

    def _set_results_dict(self):

        self._results_mp = {}
        self._results_i = {}
        self._results_w = {}
        self._results_v = {}
        self._results_mpl = {}

    def _process(self, id, image):

        image_width, image_height, _ = image.shape

        if self._results_mp[id].multi_face_landmarks is not None:

            self.n_landmark = len(self._results_mp[id].multi_face_landmarks[0].landmark)

            self._results_mpl[id] = self._results_mp[id].multi_face_landmarks
            self._results_i[id] = self._process_mp_to_image(id=id, image_width=image_width, image_height=image_height)
            self._results_w[id] = self._process_mp_to_w(id=id)
            self._results_v[id] = self._process_mp_to_visibility(id=id)

        else:

            self._results_i[id] = None
            self._results_w[id] = None
            self._results_v[id] = None

    def _process_mp_to_image(self, id, image_width, image_height):

        results_i = np.array([[p.x * image_width, p.y * image_height] for p
                              in self._results_mp[id].multi_face_landmarks[0].landmark]).astype(dtype=int)

        return results_i

    def _process_mp_to_w(self, id):

        landmarks = self._results_mp[id].multi_face_landmarks[0].landmark
        results_w = np.array([[-landmarks[k].z, landmarks[k].x, -landmarks[k].y] for k in range(self.n_landmark)])

        return results_w

    def _process_mp_to_visibility(self, id):

        landmarks = self._results_mp[id].multi_face_landmarks[0].landmark
        result_visibility = np.array([[landmarks[k].visibility] for k in range(self.n_landmark)])

        return result_visibility

    # endregion

    # region property

    @property
    def mp_face_mesh(self):
        return self._mp_face_mesh

    @property
    def results_i(self):
        return self._results_i

    @property
    def results_w(self):
        return self._results_w

    @property
    def results_v(self):
        return self._results_v

    @property
    def results_mpl(self):
        return self._results_mpl

    # endregion


class HolisticFrame():

    def __init__(self):

        self._mp_holistic = mp.solutions.holistic
        self.holistic_detection_parameters = dict(static_image_mode=False,
                                                  min_detection_confidence=0.5,
                                                  model_complexity=1,
                                                  min_tracking_confidence=0.2,
                                                  smooth_landmarks=True)

    def infer(self, image, person_results, ids_results):

        self._set_results_dict()
        MARGIN = 10

        for i, pose in enumerate(person_results):

            id = ids_results[i]

            xmin = pose[0]
            xmax = pose[2]
            ymin = pose[1]
            ymax = pose[3]

            img = image[(int(ymin) - MARGIN):(int(ymax) + MARGIN), (int(xmin) - MARGIN):(int(xmax) + MARGIN)]

            with self._mp_holistic.Holistic(**self.holistic_detection_parameters) as holistic:

                self._results_mp[id] = holistic.process(img)

                self._process(id=id, image=image)

        return self._results_mp

    # region To process the mp results

    def _set_results_dict(self):

        self._results_mp = {}
        self._results_i = {}
        self._results_w = {}
        self._results_v = {}
        self._results_mpl = {}

    def _process(self, id, image):

        image_width, image_height, _ = image.shape

        if self._results_mp[id].pose_landmarks:

            self.n_landmark = len(self._results_mp[id].pose_landmarks.landmark)

            self._results_mpl[id] = self._results_mp[id].pose_landmarks
            self._results_i[id] = self._process_mp_to_image(id=id, image_width=image_width, image_height=image_height)
            self._results_w[id] = self._process_mp_to_w(id=id)
            self._results_v[id] = self._process_mp_to_visibility(id=id)

        else:

            self._results_mpl[id] = None
            self._results_i[id] = None
            self._results_w[id] = None
            self._results_v[id] = None

    def _process_mp_to_image(self, id, image_width, image_height):

        results_i = np.array([[p.x * image_width, p.y * image_height] for p
                              in self._results_mp[id].pose_landmarks.landmark]).astype(dtype=int).tolist()

        return results_i

    def _process_mp_to_w(self, id):

        landmarks = self._results_mp[id].pose_landmarks.landmark
        results_w = np.array([[-landmarks[k].z, landmarks[k].x, -landmarks[k].y] for k in range(self.n_landmark)]).tolist()

        return results_w

    def _process_mp_to_visibility(self, id):

        landmarks = self._results_mp[id].pose_landmarks.landmark
        result_visibility = np.array([[landmarks[k].visibility] for k in range(self.n_landmark)]).tolist()

        return result_visibility

    # endregion

    # region property

    @property
    def mp_holistic(self):
        return self._mp_holistic

    @property
    def results_i(self):
        return self._results_i

    @property
    def results_w(self):
        return self._results_w

    @property
    def results_v(self):
        return self._results_v

    @property
    def results_mpl(self):
        return self._results_mpl

    # endregion

