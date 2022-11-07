import copy
from Classes.ProcessWebcam import *
import cv2
from ClassesML.ModelMLUtilities import *
from ClassesDB.DatabaseIDs import *
from retinaface import RetinaFace
import copy

class FaceDetectionModel:

    def __init__(self, path_model, model_name):

        self.device = "CPU"
        self.num_requests = 2

        self.model, self.compiled_model = ModelMLUtilities.load_local_xml_model(path_model + model_name, device_name=self.device)

        ModelMLUtilities.display_input_onnx_model_information(self.model)
        ModelMLUtilities.display_output_onnx_model_information(self.model)

        self.input_name = ModelMLUtilities.get_input_name_onnx_model(self.model)
        self.output_name = ModelMLUtilities.get_output_name_onnx_model(self.model)

        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)

        self.input_size = self.compiled_model.input(0).shape
        self.output_size = self.compiled_model.output(0).shape

        self.input_height = self.input_size[2]
        self.input_width = self.input_size[3]

    def scale_frame(self, frame):

        initial_h, initial_w = frame.shape[:2]
        scale_h, scale_w = initial_h / float(self.input_height), initial_w / float(self.input_width)
        in_frame = cv2.resize(frame, (self.input_width, self.input_height))
        in_frame = in_frame.transpose((2, 0, 1))
        in_frame = in_frame.reshape(self.input_size)

        return in_frame, scale_h, scale_w

class PersonFaceDetectorOV(FaceDetectionModel):

    def __init__(self, path_model, threshold: float = 0.6):

        self.model_name = "face-detection-retail-0005"
        self.threshold = threshold
        self._conf = []

        super().__init__(path_model=path_model, model_name=self.model_name)

    def infer(self, image):

        scaled_frame, scale_h, scale_w = self.scale_frame(frame=image)

        infer_result = self.compiled_model([scaled_frame])[self.output_layer]

        detections = self._get_detection(infer_result=infer_result, image=image)

        persons = self._get_persons(detections=detections)

        return detections, persons

    def _get_detection(self, infer_result, image):

        detections = []
        height, width = image.shape[:2]
        for r in infer_result[0][0]:
            conf = r[2]
            if (conf > self.threshold):
                x1 = int(r[3] * width)
                y1 = int(r[4] * height)
                x2 = int(r[5] * width)
                y2 = int(r[6] * height)
                detections.append([x1, y1, x2, y2, conf])

        return detections

    def _get_persons(self, detections):

        persons = []
        self._conf = []

        if (len(detections) > 0):

            print("--------- faces: {} ---------".format(len(detections)))
            for detection in detections:
                conf = detection[4]
                if (conf > self.threshold):

                    x1 = int(detection[0])
                    y1 = int(detection[1])
                    x2 = int(detection[2])
                    y2 = int(detection[3])

                    print("{:.1f} ({},{})-({},{})".format(conf, x1, y1, x2, y2))
                    persons.append([x1, y1, x2, y2])
                    self._conf.append(float(conf))

        return persons

    @property
    def conf(self):
        return self._conf

class PersonFaceDetectorMP:

    def __init__(self, path_model: str = "", threshold: float = 0.6):

        self.path_model = path_model
        self.model_name = "media_pipe"
        self.threshold = threshold

        self.mp_face_detection = mp.solutions.face_detection
        self._conf = []

        """
        FACE_DETECTION_PARAMETERS: 
        model_selection: This argument takes the real integer value only in the range of 0-1 i.e. this model will take the integer value as either 1 or 0. Let us discuss these two types of models.
            - 0 type model: When we will select the 0 type model then our face detection model will be able to detect the faces within the range of 2 meters from the camera.
            - 1 type model: When we will select the 1 type model then our face detection model will be able to detect the faces within the range of 5 meters. Though the default value is 0.
        min_detection_confidence: This argument also takes the integer value but in the range of [0.0,1.0] and the default value for the same is 0.5 which is 50% confidence i.e. when our model will be detecting the faces it should be at least 50% sure that the face is there otherwise it won’t detect anything.
        """

        self.face_detection_parameters = dict(model_selection=1,
                                              min_detection_confidence=self.threshold)

    def infer(self, image):

        img = copy.deepcopy(image)

        detections = self._get_detection(image=img)

        persons = self._get_persons(detections=detections)

        return detections, persons

    def _get_persons(self, detections):

        persons = []
        self._conf = []

        if detections:
            if (len(detections) > 0):
                print("--------- faces: {} ---------".format(len(detections)))
                for n in range(len(detections)):

                    detection = detections[n]
                    temp = detection.location_data.relative_bounding_box
                    conf = detection.score[0]

                    box_xmin = int(temp.xmin * self.image_width)
                    box_ymin = int(temp.ymin * self.image_height)
                    box_xmax = int(box_xmin + temp.width * self.image_width)
                    box_ymax = int(box_ymin + temp.height * self.image_height)
                    box = [box_xmin, box_ymin, box_xmax, box_ymax]

                    print("Conf - {:.1f}, Location - ({}, {})-({}, {})".format(conf, box_xmin, box_ymin, box_xmax, box_ymax))
                    self._conf.append(float(conf))
                    persons.append(box)

        else:
            self._conf.append(None)

        return persons

    def _get_detection(self, image):

        with self.mp_face_detection.FaceDetection(**self.face_detection_parameters) as face_detection:
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.image_height, self.image_width, _ = image.shape

            results_temp = face_detection.process(image)
            detections = results_temp.detections

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        return detections

    @property
    def conf(self):
        return self._conf


class PersonFaceDetectorRF():

    def __init__(self, path_model: str = "", threshold: float = 0.6):

        self.path_model = path_model
        self.model_name = "retina_face"
        self.threshold = threshold

        # initialize the list of confidence for each face
        self._conf = []

    def infer(self, image):

        img = copy.deepcopy(image)

        detections = self._get_detection(image=img)

        persons = self._get_persons(detections=detections)

        return detections, persons

    def _get_persons(self, detections):

        persons = []
        self._conf = []

        if detections:
            print("--------- faces: {} ---------".format(len(detections)))
            for key in detections:

                identify = detections[key]
                facial_area = identify["facial_area"]
                conf = identify["score"]

                # the facial are is already int type
                box_xmin = int(facial_area[0])
                box_ymin = int(facial_area[1])
                box_xmax = int(facial_area[2])
                box_ymax = int(facial_area[3])

                box = [box_xmin, box_ymin, box_xmax, box_ymax]
                print("Conf - {:.1f}, Location - ({}, {})-({}, {})".format(conf, box_xmin, box_ymin, box_xmax, box_ymax))
                self._conf.append(float(conf))
                persons.append(box)

        else:
            self._conf = [None]

        return persons

    def _get_detection(self, image):

        detections = RetinaFace.detect_faces(image, threshold=self.threshold)

        return detections

    @property
    def conf(self):
        return self._conf