import pandas as pd
import itertools
from tabulate import tabulate
from enum import Enum

class DetectionType(Enum):

    BODY_BOX_YOLO = 1
    BODY_BOX_OV = 2

    FACE_BOX_OV = 3
    FACE_BOX_MP = 4  # Mediapipe (use SSD)
    FACE_BOX_RF = 5

class HyperParameters:

    def __init__(self, hyper_dict: dict):

        self.hyper_dict = hyper_dict

        # Which detection
        self.detection_type = hyper_dict["detection_type"]

        # Part related to body or face detection model
        self.confidence_threshold = hyper_dict["confidence_threshold"]

        # Part related to ID identification database
        self.DETECTION_THRESHOLD_UP = hyper_dict["DETECTION_THRESHOLD_UP"]
        self.DETECTION_THRESHOLD_DOWN = hyper_dict["DETECTION_THRESHOLD_DOWN"]
        self.DISTANCE_THRESHOLD = hyper_dict["DISTANCE_THRESHOLD"]
        self.DETECTION_N = hyper_dict["DETECTION_N"]

class HyperParametersGenerator:

    def __init__(self, detection_type: DetectionType):

        self.detection_type = detection_type

    def generate(self):

        display_info = True

        hyperparameters_choices = dict(detection_type=[DetectionType.BODY_BOX_YOLO, DetectionType.BODY_BOX_OV,
                                                       DetectionType.FACE_BOX_OV, DetectionType.FACE_BOX_MP,
                                                       DetectionType.FACE_BOX_RF],
                                       confidence_threshold=[0.4, 0.6, 0.8],
                                       DETECTION_THRESHOLD_UP=[0.90, 0.95],
                                       DETECTION_THRESHOLD_DOWN=[0.4, 0.5],
                                       DISTANCE_THRESHOLD=[50, 100],
                                       DETECTION_N=[3, 5])

        keys, values = zip(*hyperparameters_choices.items())
        self.hyperparameters_all_combination = [dict(zip(keys, v)) for v in itertools.product(*values)]

        df = pd.DataFrame.from_dict(self.hyperparameters_all_combination)

        if display_info:
            print(tabulate(df, headers='keys', tablefmt='psql'))

        self.n_hyper = len(self.hyperparameters_all_combination)

        return self.hyperparameters_all_combination

    def set_scores(self, scores: list):

        self.scores = scores
        display_info = True

        if self.n_hyper == len(self.scores):

            for i in range(self.n_hyper):
                self.hyperparameters_all_combination[i]["score"] = self.scores[i]

        df = pd.DataFrame.from_dict(self.hyperparameters_all_combination)

        if display_info:
            print(tabulate(df, headers='keys', tablefmt='psql'))

    def save(self, path: str = ""):

        df = pd.DataFrame.from_dict(self.hyperparameters_all_combination)

        name = "search.csv"
        df.to_csv(path + name)

        print('hyperparameters_all_combination has been saved to: ' + path + name)

    def load(self, path: str = ""):

        name = "search.csv"
        self.hyperparameters_all_combination = pd.read_csv(path + name)
        print('hyperparameters_all_combination has been load from:  ' + path + name)

    @staticmethod
    def get_best_hyper_each_detection_type():

        hyper_combination_best_dict = {}

        # Best hyperparameter so far for each model
        hyper_combination_best_dict[DetectionType.BODY_BOX_YOLO] = {'detection_type': DetectionType.BODY_BOX_YOLO,
                                                                    'confidence_threshold': 0.4,
                                                                    'DETECTION_THRESHOLD_UP': 0.95,
                                                                    'DETECTION_THRESHOLD_DOWN': 0.5,
                                                                    'DISTANCE_THRESHOLD': 100,
                                                                    'DETECTION_N': 5}

        hyper_combination_best_dict[DetectionType.BODY_BOX_OV] = {'detection_type': DetectionType.BODY_BOX_OV,
                                                                  'confidence_threshold': 0.4,
                                                                  'DETECTION_THRESHOLD_UP': 0.95,
                                                                  'DETECTION_THRESHOLD_DOWN': 0.5,
                                                                  'DISTANCE_THRESHOLD': 100,
                                                                  'DETECTION_N': 5}

        hyper_combination_best_dict[DetectionType.FACE_BOX_OV] = {'detection_type': DetectionType.FACE_BOX_OV,
                                                                  'confidence_threshold': 0.4,
                                                                  'DETECTION_THRESHOLD_UP': 0.95,
                                                                  'DETECTION_THRESHOLD_DOWN': 0.5,
                                                                  'DISTANCE_THRESHOLD': 100,
                                                                  'DETECTION_N': 5}

        hyper_combination_best_dict[DetectionType.FACE_BOX_MP] = {'detection_type': DetectionType.FACE_BOX_MP,
                                                                  'confidence_threshold': 0.4,
                                                                  'DETECTION_THRESHOLD_UP': 0.95,
                                                                  'DETECTION_THRESHOLD_DOWN': 0.5,
                                                                  'DISTANCE_THRESHOLD': 100,
                                                                  'DETECTION_N': 5}

        hyper_combination_best_dict[DetectionType.FACE_BOX_RF] = {'detection_type': DetectionType.FACE_BOX_RF,
                                                                  'confidence_threshold': 0.4,
                                                                  'DETECTION_THRESHOLD_UP': 0.95,
                                                                  'DETECTION_THRESHOLD_DOWN': 0.5,
                                                                  'DISTANCE_THRESHOLD': 100,
                                                                  'DETECTION_N': 5}
        return hyper_combination_best_dict

    @staticmethod
    def get_best_hyper():

        hyper_combination_best_dict = {}

        # Best hyperparameter so far for each model
        hyper_combination_best_dict[DetectionType.BODY_BOX_YOLO] = {'detection_type': DetectionType.BODY_BOX_YOLO,
                                                                    'confidence_threshold': 0.4,
                                                                    'DETECTION_THRESHOLD_UP': 0.95,
                                                                    'DETECTION_THRESHOLD_DOWN': 0.5,
                                                                    'DISTANCE_THRESHOLD': 100,
                                                                    'DETECTION_N': 5}

        hyper_combination_best_dict[DetectionType.FACE_BOX_OV] = {'detection_type': DetectionType.FACE_BOX_OV,
                                                                  'confidence_threshold': 0.4,
                                                                  'DETECTION_THRESHOLD_UP': 0.95,
                                                                  'DETECTION_THRESHOLD_DOWN': 0.5,
                                                                  'DISTANCE_THRESHOLD': 100,
                                                                  'DETECTION_N': 5}

        return hyper_combination_best_dict
