import numpy as np
import pandas as pd
import math

class Euler():

    @staticmethod
    def rad2deg(rad):

        if isinstance(rad, list):
            rad = np.array(rad)

        else:
            rad = rad

        deg = rad / np.pi * 180

        return deg

    @staticmethod
    def deg2rad(deg):

        if isinstance(deg, list):
            deg = np.array(deg)

        else:
            deg = deg

        rad = deg / 180 * np.pi

        return rad

    @staticmethod
    def rotationX(angle):

        R = np.array([[1,                0,                               0],
                      [0,                math.cos(angle),  -math.sin(angle)],
                      [0,                math.sin(angle),   math.cos(angle)]])

        return R

    @staticmethod
    def rotationY(angle):

        R = np.array([[math.cos(angle),    0,                  -math.sin(angle)],
                      [0,                  1,                                 0],
                      [math.sin(angle),    0,                   math.cos(angle)]])

        return R

    @staticmethod
    def rotationZ(angle):

        R = np.array([[math.cos(angle),   -math.sin(angle),                 0],
                      [math.sin(angle),   math.cos(angle),                  0],
                      [0,                 0,                                1]])

        return R

    @staticmethod
    def euler_angles_to_rotation_matrix(euler, seq="XYZ", to_rad=True):

        if to_rad:

            euler = Euler.deg2rad(euler)

        if seq == "XYZ":

            R = Euler.rotationX(euler[0]) @ Euler.rotationY(euler[1]) @ Euler.rotationZ(euler[2])

        elif seq == "XZY":

            R = Euler.rotationX(euler[0]) @ Euler.rotationZ(euler[1]) @ Euler.rotationY(euler[2])

        elif seq == "YXZ":

            R = Euler.rotationY(euler[0]) @ Euler.rotationX(euler[1]) @ Euler.rotationZ(euler[2])

        elif seq == "YZX":

            R = Euler.rotationY(euler[0]) @ Euler.rotationZ(euler[1]) @ Euler.rotationX(euler[2])

        elif seq == "ZXY":

            R = Euler.rotationZ(euler[0]) @ Euler.rotationX(euler[1]) @ Euler.rotationY(euler[2])

        elif seq == "ZYX":

            R = Euler.rotationZ(euler[0]) @ Euler.rotationY(euler[1]) @ Euler.rotationX(euler[2])

        return R

class Rotation():

    @staticmethod
    def rotation_matrix_to_euler_angles(rotation_matrix, seq="XYZ", to_degrees:bool = True):

        if seq == "XYZ":

            beta = np.arctan2(rotation_matrix[0, 2],
                              np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[0, 1] ** 2))

            alpha = np.arctan2(-rotation_matrix[1, 2] / np.cos(beta), rotation_matrix[2, 2] / np.cos(beta))

            gamma = np.arctan2(-rotation_matrix[0, 1] / np.cos(beta), rotation_matrix[0, 0] / np.cos(beta))

        elif seq == "XZY":

            beta = np.arctan2(-rotation_matrix[0, 1],
                              np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[0, 2] ** 2))

            alpha = np.arctan2(rotation_matrix[2, 1] / np.cos(beta), rotation_matrix[1, 1] / np.cos(beta))

            gamma = np.arctan2(rotation_matrix[0, 2] / np.cos(beta), rotation_matrix[0, 0] / np.cos(beta))

        elif seq == "ZXY":

            beta = np.arctan2(rotation_matrix[2, 1],
                              np.sqrt(rotation_matrix[2, 0] ** 2 + rotation_matrix[2, 2] ** 2))

            alpha = np.arctan2(-rotation_matrix[0, 1] / np.cos(beta), rotation_matrix[1, 1] / np.cos(beta))

            gamma = np.arctan2(-rotation_matrix[2, 0] / np.cos(beta), rotation_matrix[2, 2] / np.cos(beta))

        elif seq == "ZYX":

            beta = np.arctan2(-rotation_matrix[2, 0],
                              np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2))

            alpha = np.arctan2(rotation_matrix[1, 0] / np.cos(beta), rotation_matrix[0, 0] / np.cos(beta))

            gamma = np.arctan2(rotation_matrix[2, 1] / np.cos(beta), rotation_matrix[2, 2] / np.cos(beta))

        euler = np.array([alpha, beta, gamma])

        if to_degrees:
            euler = Euler.rad2deg(euler)

        return euler


    @staticmethod
    def compute_relative_rotation_matrix(parent, child):

        rotation_matrix = np.dot(child, np.linalg.inv(parent))

        return rotation_matrix

class PoseFeatures():

    @staticmethod
    def compute_trunk_reference_system(left_shoulder, right_shoulder, left_hip, right_hip):

        O = ((left_shoulder + right_shoulder) * .5 + (left_hip + right_hip) * .5) * .5

        Y = (left_shoulder + right_shoulder) * .5 - (left_hip + right_hip) * .5
        Y_norm = np.linalg.norm(Y)
        Y = Y / Y_norm

        vec_temp = (left_hip - right_hip)
        vec_temp_norm = np.linalg.norm(vec_temp)
        vec_temp = vec_temp / vec_temp_norm

        Z = np.cross(vec_temp, Y)
        Z_norm = np.linalg.norm(Z)
        Z = Z / Z_norm

        X = np.cross(Y, Z)
        X_norm = np.linalg.norm(X)
        X = X / X_norm

        trunk_reference_system = np.hstack([X.reshape(-1, 1), Y.reshape(-1, 1), Z.reshape(-1, 1)])

        return O, trunk_reference_system

    @staticmethod
    def compute_head_reference_system(nose, left_eye_inner, right_eye_inner, left_ear, right_ear, mouth_left, mouth_right):

        O = nose

        Y = (left_eye_inner + right_eye_inner) * .5 - (mouth_left + mouth_right) * .5
        Y_norm = np.linalg.norm(Y)
        Y = Y / Y_norm

        vec_temp = (left_ear - right_ear)
        vec_temp_norm = np.linalg.norm(vec_temp)
        vec_temp = vec_temp / vec_temp_norm

        Z = np.cross(vec_temp, Y)
        Z_norm = np.linalg.norm(Z)
        Z = Z / Z_norm

        X = np.cross(Y, Z)
        X_norm = np.linalg.norm(X)
        X = X / X_norm

        reference_system = np.hstack([X.reshape(-1, 1), Y.reshape(-1, 1), Z.reshape(-1, 1)])

        return O, reference_system

    @staticmethod
    def compute_articulation_angle(lm1, lm2, lm3):

        if isinstance(lm1, list):
            lm1 = np.array(lm1)
        if isinstance(lm2, list):
            lm2 = np.array(lm2)
        if isinstance(lm3, list):
            lm3 = np.array(lm3)

        vector_a = lm2 - lm1
        vector_b = lm2 - lm3

        angle = np.arccos(np.dot(vector_a, vector_b) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b)))
        # angle[np.isnan(angle)] = 0
        angle = angle / np.pi * 180

        return angle

    @staticmethod
    def get_origin_w():

        x = np.array([1,  0,  0])
        y = np.array([0,  1,  0])
        z = np.array([0,  0,  1])

        origin_reference_system = np.hstack([x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)])

        return origin_reference_system

class PresenceVisibility():

    def __init__(self, results_v, threshold: float = 0.8):

        self._results_v = results_v
        self.threshold = threshold

        self._filter_data()

    def _filter_data(self):

        # nose
        self._v_nose = self._results_v[0]

        # eye
        self._v_eye_inner_l = self._results_v[1]
        self._v_eye_l = self._results_v[2]
        self._v_eye_outer_l = self._results_v[3]

        self._v_eye_inner_r = self._results_v[4]
        self._v_eye_r = self._results_v[5]
        self._v_eye_outer_r = self._results_v[6]

        # ear
        self._v_ear_l = self._results_v[7]
        self._v_ear_r = self._results_v[8]

        # mouth
        self._v_mouth_l = self._results_v[9]
        self._v_mouth_r = self._results_v[10]

        # glenohumeral joint
        self._v_glenohumeral_l = self._results_v[11]
        self._v_glenohumeral_r = self._results_v[12]

        # elbow
        self._v_elbow_l = self._results_v[13]
        self._v_elbow_r = self._results_v[14]

        # wrist
        self._v_wrist_l = self._results_v[15]
        self.v_wrist_r = self._results_v[16]

        # pinky
        self._v_pinky_l = self._results_v[17]
        self.v_pinky_r = self._results_v[18]

        # index
        self._v_index_l = self._results_v[19]
        self._v_index_r = self._results_v[20]

        # thumb
        self._v_thumb_l = self._results_v[21]
        self._v_thumb_r = self._results_v[22]

        # hip
        self._v_hip_l = self._results_v[23]
        self._v_hip_r = self._results_v[24]

        # knee
        self._v_knee_l = self._results_v[25]
        self._v_knee_r = self._results_v[26]

        # ankle
        self._v_ankle_l = self._results_v[27]
        self._v_ankle_r = self._results_v[28]

        # heel
        self._v_heel_l = self._results_v[29]
        self._v_heel_r = self._results_v[30]

        # foot
        self._v_foot_l = self._results_v[31]
        self._v_foot_r = self._results_v[32]

    def check_presence_trunk_reference_system(self):

        is_reference_system = False

        if (
                self._v_glenohumeral_l > self.threshold
                and self._v_glenohumeral_r > self.threshold
                and self._v_hip_l > self.threshold
                and self._v_hip_r > self.threshold
        ):

            is_reference_system = True

        return is_reference_system

    def check_presence_head_reference_system(self):

        is_reference_system = False

        if (
                self._v_nose > self.threshold
                and self._v_eye_inner_l > self.threshold
                and self._v_eye_inner_r > self.threshold
                and self._v_ear_l > self.threshold
                and self._v_ear_r > self.threshold
                and self._v_mouth_l > self.threshold
                and self._v_mouth_r > self.threshold
        ):


            is_reference_system = True

        return is_reference_system

    def check_presence_elbows_angles(self):

        is_presence_l = False

        if (
                self._v_glenohumeral_l > self.threshold
                and self._v_elbow_l > self.threshold
                and self._v_wrist_l > self.threshold
        ):

            is_presence_l = True

        is_presence_r = False

        if (
                self._v_glenohumeral_r > self.threshold
                and self._v_elbow_r > self.threshold
                and self.v_wrist_r > self.threshold
        ):

            is_presence_r = True

        return is_presence_l, is_presence_r

    def check_presence_glenohumeral_angles(self):

        is_presence_l = False

        if (
                self._v_elbow_l > self.threshold
                and self._v_glenohumeral_l > self.threshold
                and self._v_hip_l > self.threshold
        ):

            is_presence_l = True

        is_presence_r = False

        if (
                self._v_elbow_r > self.threshold
                and self._v_glenohumeral_r > self.threshold
                and self._v_hip_r > self.threshold
        ):

            is_presence_r = True

        return is_presence_l, is_presence_r

    def check_presence_wrists_angles(self):

        is_presence_l = False

        if (
                self._v_pinky_l > self.threshold
                and self._v_index_l > self.threshold
                and self._v_elbow_l > self.threshold
                and self._v_wrist_l > self.threshold
        ):

            is_presence_l = True

        is_presence_r = False

        if (
                self.v_pinky_r > self.threshold
                and self._v_index_r > self.threshold
                and self._v_elbow_r > self.threshold
                and self.v_wrist_r > self.threshold
        ):

            is_presence_r = True

        return is_presence_l, is_presence_r

    def check_presence_hip_angles(self):

        is_presence_l = False

        if (
                self._v_hip_r > self.threshold
                and self._v_hip_l > self.threshold
                and self._v_knee_l > self.threshold
        ):

            is_presence_l = True

        is_presence_r = False

        if (self._v_hip_l > self.threshold and self._v_hip_r > self.threshold and self._v_knee_r > self.threshold):

            is_presence_r = True

        return is_presence_l, is_presence_r

    def check_presence_knees_angles(self):

        is_presence_l = False

        if (self._v_hip_l > self.threshold and self._v_knee_l > self.threshold and self._v_ankle_l > self.threshold):

            is_presence_l = True

        is_presence_r = False

        if (self._v_hip_r > self.threshold and self._v_knee_r > self.threshold and self._v_ankle_r > self.threshold):

            is_presence_r = True

        return is_presence_l, is_presence_r

    def check_presence_ankle_angles(self):

        is_presence_l = False

        if (self._v_knee_l > self.threshold and self._v_ankle_l > self.threshold and self._v_foot_l > self.threshold):

            is_presence_l = True

        is_presence_r = False

        if (self._v_knee_r > self.threshold and self._v_ankle_r > self.threshold and self._v_foot_r > self.threshold):

            is_presence_r = True

        return is_presence_l, is_presence_r

class BiomechanicsImageInformation():

    def __init__(self, results_i):

        self.results_i = results_i
        self._dict = {}
        self._retract_image_position()

    @property
    def dict(self):
        return self._dict

    def _retract_image_position(self):

        self._dict["elbow_l"] = self.results_i[13]
        self._dict["elbow_r"] = self.results_i[14]

        self._dict["glenohumeral_l"] = self.results_i[11]
        self._dict["glenohumeral_r"] = self.results_i[12]

        self._dict["wrist_l"] = self.results_i[15]
        self._dict["wrist_r"] = self.results_i[16]

        self._dict["hip_l"] = self.results_i[23]
        self._dict["hip_r"] = self.results_i[24]

        self._dict["knee_l"] = self.results_i[25]
        self._dict["knee_r"] = self.results_i[26]

        self._dict["ankle_l"] = self.results_i[27]
        self._dict["ankle_r"] = self.results_i[28]

class BiomechanicsInformation():

    def __init__(self, results_hollistic_w, presenceVisibility: PresenceVisibility, display = True):

        self.presenceVisibility = presenceVisibility

        self._results_w = results_hollistic_w
        self.display = display

        self._filter_data()

        self._dict = {}

        self._retract_angles()

    @property
    def dict(self):
        return self._dict

    def _retract_angles(self):

        self._dict["elbow_l"], \
        self._dict["elbow_r"] = self._compute_elbows_angles()
        # print("Light elbow articulation angle: " + str(elbow_angle_l))
        self._dict["glenohumeral_l"], \
        self._dict["glenohumeral_r"] = self._compute_glenohumeral_angles()

        self._dict["wrist_l"], \
        self._dict["wrist_r"] = self._compute_wrists_angles()

        self._dict["hip_l"], \
        self._dict["hip_r"] = self._compute_hip_angles()

        self._dict["knee_l"], \
        self._dict["knee_r"] = self._compute_knees_angles()

        self._dict["ankle_l"], \
        self._dict["ankle_r"] = self._compute_ankles_angles()

        self._dict["trunk_roll"], \
        self._dict["trunk_pitch"], \
        self._dict["trunk_yaw"] = self._compute_trunk_rotation_matrix_to_euler()

        if self.display:

            print(self._dict.items())

    def _filter_data(self):

        self._lm_glenohumeral_l = self._results_w[11]
        self._lm_glenohumeral_r = self._results_w[12]

        self._lm_elbow_l = self._results_w[13]
        self._lm_elbow_r = self._results_w[14]

        self._lm_wrist_l = self._results_w[15]
        self.lm_wrist_r = self._results_w[16]

        self._lm_pinky_l = self._results_w[17]
        self.lm_pinky_r = self._results_w[18]

        self._lm_index_l = self._results_w[19]
        self._lm_index_r = self._results_w[20]

        self._lm_hip_l = self._results_w[23]
        self._lm_hip_r = self._results_w[24]

        self._lm_knee_l = self._results_w[25]
        self._lm_knee_r = self._results_w[26]

        self._lm_ankle_l = self._results_w[27]
        self._lm_ankle_r = self._results_w[28]

        self._lm_foot_l = self._results_w[31]
        self._lm_foot_r = self._results_w[32]

    def _compute_elbows_angles(self):

        is_l, is_r = self.presenceVisibility.check_presence_elbows_angles()

        if (is_l):

            elbow_angle_l = PoseFeatures.compute_articulation_angle(lm1=self._lm_glenohumeral_l,
                                                                    lm2=self._lm_elbow_l,
                                                                    lm3=self._lm_wrist_l)

        else:

            elbow_angle_l = None

        if (is_r):

            elbow_angle_r = PoseFeatures.compute_articulation_angle(lm1=self._lm_glenohumeral_r,
                                                                    lm2=self._lm_elbow_r,
                                                                    lm3=self.lm_wrist_r)

        else:

            elbow_angle_r = None

        return elbow_angle_l, elbow_angle_r

    def _compute_glenohumeral_angles(self):

        is_l, is_r = self.presenceVisibility.check_presence_glenohumeral_angles()

        if (is_l):

            glenohumeral_angle_l = PoseFeatures.compute_articulation_angle(lm1=self._lm_elbow_l,
                                                                           lm2=self._lm_glenohumeral_l,
                                                                           lm3=self._lm_hip_l)

        else:

            glenohumeral_angle_l = None

        if(is_r):

            glenohumeral_angle_r = PoseFeatures.compute_articulation_angle(lm1=self._lm_elbow_r,
                                                                           lm2=self._lm_glenohumeral_r,
                                                                           lm3=self._lm_hip_r)

        else:

            glenohumeral_angle_r = None

        return glenohumeral_angle_l, glenohumeral_angle_r

    def _compute_wrists_angles(self):

        is_l, is_r = self.presenceVisibility.check_presence_wrists_angles()

        if (is_l):

            temp_l = (self._lm_pinky_l + self._lm_index_l) * .5

            wrist_angle_l = PoseFeatures.compute_articulation_angle(lm1=self._lm_elbow_l,
                                                                    lm2=self._lm_wrist_l,
                                                                    lm3=temp_l)

        else:

            wrist_angle_l = None

        if (is_r):

            temp_r = (self.lm_pinky_r + self._lm_index_r) * .5

            wrist_angle_r = PoseFeatures.compute_articulation_angle(lm1=self._lm_elbow_r,
                                                                    lm2=self.lm_wrist_r,
                                                                    lm3=temp_r)

        else:

            wrist_angle_r = None

        return wrist_angle_l, wrist_angle_r

    def _compute_hip_angles(self):

        is_l, is_r = self.presenceVisibility.check_presence_hip_angles()

        if (is_l):

            hip_angle_l = PoseFeatures.compute_articulation_angle(lm1=self._lm_hip_r,
                                                                  lm2=self._lm_hip_l,
                                                                  lm3=self._lm_knee_l)

        else:

            hip_angle_l = None

        if (is_r):

            hip_angle_r = PoseFeatures.compute_articulation_angle(lm1=self._lm_hip_l,
                                                                  lm2=self._lm_hip_r,
                                                                  lm3=self._lm_knee_r)

        else:

            hip_angle_r = None

        return hip_angle_l, hip_angle_r

    def _compute_knees_angles(self):

        is_l, is_r = self.presenceVisibility.check_presence_knees_angles()

        if(is_l):

            knee_angle_l = PoseFeatures.compute_articulation_angle(lm1=self._lm_hip_l,
                                                                   lm2=self._lm_knee_l,
                                                                   lm3=self._lm_ankle_l)

        else:

            knee_angle_l = None

        if(is_r):

            knee_angle_r = PoseFeatures.compute_articulation_angle(lm1=self._lm_hip_r,
                                                                   lm2=self._lm_knee_r,
                                                                   lm3=self._lm_ankle_r)

        else:

            knee_angle_r = None

        return knee_angle_l, knee_angle_r

    def _compute_ankles_angles(self):

        is_l, is_r = self.presenceVisibility.check_presence_ankle_angles()

        if (is_l):

            ankle_angle_l = PoseFeatures.compute_articulation_angle(lm1=self._lm_knee_l,
                                                                    lm2=self._lm_ankle_l,
                                                                    lm3=self._lm_foot_l)

        else:

            ankle_angle_l = None

        if (is_r):

            ankle_angle_r = PoseFeatures.compute_articulation_angle(lm1=self._lm_knee_r,
                                                                    lm2=self._lm_ankle_r,
                                                                    lm3=self._lm_foot_r)

        else:

            ankle_angle_r = None

        return ankle_angle_l, ankle_angle_r

    def _compute_trunk_rotation_matrix_to_euler(self):

        O, trunk_reference_system = PoseFeatures.compute_trunk_reference_system(left_shoulder=self._lm_glenohumeral_l,
                                                                                right_shoulder=self._lm_glenohumeral_r,
                                                                                left_hip=self._lm_hip_l,
                                                                                right_hip=self._lm_hip_r)
        # In world coordination system, reference system is equal to rotation matrix
        # reference: https://learnopencv.com/rotation-matrix-to-euler-angles/
        euler = Rotation.rotation_matrix_to_euler_angles(rotation_matrix=trunk_reference_system,
                                                         seq="ZYX",
                                                         to_degrees=True)

        return euler[0], euler[1], euler[2]
