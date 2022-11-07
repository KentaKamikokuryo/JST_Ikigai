from abc import ABC, abstractmethod
from Classes.Drawing import Drawing
from ClassesML.BodyDetectionModel import PersonBodyDetectorOV, PersonBodyDetectionYOLO
from ClassesML.FaceDetectionModel import PersonFaceDetectorOV, PersonFaceDetectorMP
from ClassesML.FaceEmotionDetectionModel import FaceEmotionRecognition
from Kenta.TempClassesML.FaceDetectorRetina import PersonFaceDetectorRF
from Kenta.TempClassesML.Models import DetectionType, FaceEmotionsDetectionType
import sys


class IFactory(ABC):

    @abstractmethod
    def _create(self):
        pass

    @property
    @abstractmethod
    def model_name(self):
        pass

    @property
    @abstractmethod
    def json_name(self):
        pass

    @property
    @abstractmethod
    def color(self):
        pass

    @property
    @abstractmethod
    def model(self):
        pass

    @property
    @abstractmethod
    def pos(self):
        pass


class DetectorFactory(IFactory):

    def __init__(self, detection_type: DetectionType, path_model: str, threshold: float = 0.6):

        self.detection_type = detection_type
        self.path_model = path_model
        self.threshold = threshold

        self._create()

    def _create(self):

        if self.detection_type == DetectionType.BODY_BOX_YOLO:

            self._model_name = self.detection_type
            self._json_name = "IDs"
            self._color = Drawing.color_red
            self._pos = 1
            self._model = PersonBodyDetectionYOLO(path_model=self.path_model,
                                                  threshold=self.threshold)

        elif self.detection_type == DetectionType.BODY_BOX_OV:

            self._model_name = self.detection_type
            self._json_name = "IDs"
            self._color = Drawing.color_blue
            self._pos = 2
            self._model = PersonBodyDetectorOV(path_model=self.path_model,
                                               threshold=self.threshold)

        elif self.detection_type == DetectionType.FACE_BOX_OV:

            self._model_name = self.detection_type
            self._json_name = "Face_IDs"
            self._color = Drawing.color_blue
            self._pos = 1
            self._model = PersonFaceDetectorOV(path_model=self.path_model,
                                               threshold=self.threshold)

        elif self.detection_type == DetectionType.FACE_BOX_MP:

            self._model_name = self.detection_type
            self._json_name = "Face_IDs"
            self._color = Drawing.color_red
            self._pos = 2
            self._model = PersonFaceDetectorMP(path_model=self.path_model,
                                               threshold=self.threshold)

        elif self.detection_type == DetectionType.FACE_BOX_RF:

            self._model_name = self.detection_type
            self._json_name = "Face_IDs"
            self._color = Drawing.color_green
            self._pos = 3
            self._model = PersonFaceDetectorRF(path_model=self.path_model,
                                               threshold=self.threshold)

        else:

            print("ERROR!!!!!!!!!!!!!!!!!!!!!!!")
            print("YOU HAVE TO CHECK THE DETECTION MODEL")
            sys.exit(1)

    @property
    def model_name(self):
        return self._model_name

    @property
    def json_name(self):
        return self._json_name

    @property
    def color(self):
        return self._color

    @property
    def model(self):
        return self._model

    @property
    def pos(self):
        return self._pos

class FaceEmotionsDetectorFactory(IFactory):

    def __init__(self, detection_type: FaceEmotionsDetectionType, path_model):

        self.detection_type = detection_type
        self.path_model = path_model

        self._create()

    def _create(self):

        if self.detection_type == FaceEmotionsDetectionType.EMOTIONS_OV:

            self._model_name = self.detection_type
            self._json_name = "Emotions"
            self._color = Drawing.color_blue
            self._pos = 2
            self._model = FaceEmotionRecognition(path_model=self.path_model)

        else:

            print("ERROR!!!!!!!!!!!!!!!!!!!!!!!")
            print("YOU HAVE TO CHECK THE DETECTION MODEL")
            sys.exit(1)

    @property
    def model_name(self):
        return self._model_name

    @property
    def json_name(self):
        return self._json_name

    @property
    def color(self):
        return self._color

    @property
    def model(self):
        return self._model

    @property
    def pos(self):
        return self._pos
