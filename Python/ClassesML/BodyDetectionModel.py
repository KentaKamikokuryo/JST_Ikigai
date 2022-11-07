import copy

from Classes.ProcessWebcam import *
import cv2
from ClassesML.ModelMLUtilities import *
from ClassesDB.DatabaseIDs import *

class BodyDetectionModel:

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

class PersonBodyDetectionYOLO(BodyDetectionModel):

    def __init__(self, path_model:  str, threshold: float = 0.6):

        self.model_name = "yolov5s"
        self.threshold = threshold

        # super().__init__(path_model, self.model_name)

        # Download pytorch model
        self.model = ModelMLUtilities.load_online_pytorch_model(url='ultralytics/yolov5', path_model=path_model + self.model_name)

        # since we are only interested in detecting person
        self.model.classes = [0]

        # set model parameter for confidence limit
        # conf = 0.25 - NMS confidence threshold
        # iou = 0.45 - NMS IoU threshold
        # agnostic = False - NMS class-agnostic
        # multi_label = False - NMS multiple labels per box
        # classes = None - (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
        # max_det = 1000 - maximum number of detections per image
        # amp = False - Automatic Mixed Precision (AMP) inference

        self.model.conf = self.threshold
        self.model.iou = 0.45

    def infer(self, image):

        self.image = copy.deepcopy(image)

        self.image.flags.writeable = False

        yolo_result = self._get_detection()

        self.image.flags.writeable = True

        persons = self._get_persons(results=yolo_result)

        return yolo_result, persons

    def _get_persons(self, results):

        # Get crops
        crops = results.crop(save=False)  # cropped detections dictionary

        persons = []
        self._conf = []

        if (len(crops) > 0):
            print("--------- Bodies: {} ---------".format(len(crops)))
            # Loop through all crops from YOLO
            for n in range(len(crops)):
                crop = crops[n]
                if "person" in crop["label"]:
                    box = [int(crop["box"][0]), int(crop["box"][1]), int(crop["box"][2]), int(crop["box"][3])]
                    persons.append(box)

                    conf = float(crop["conf"])
                    self._conf.append(conf)

        return persons

    def _get_detection(self):

        yolo_result = self.model(self.image)
        yolo_result.print()

        return yolo_result

    @property
    def conf(self):
        return self._conf

class PersonBodyDetectorOV(BodyDetectionModel):

    def __init__(self, path_model, threshold: float = 0.6):

        self.model_name = "person-detection-retail-0013"
        self.threshold = threshold

        super().__init__(path_model, self.model_name)

    def infer(self, image):

        self.image = image
        self.scaled_frame, self.scale_h, self.scale_w = self.scale_frame(image)

        infer_result = self.compiled_model([self.scaled_frame])[self.output_layer]

        detections = self._get_detection(infer_result)
        persons = self._get_persons(detections)

        return detections, persons

    def _get_persons(self, detections):

        persons = []
        self._conf = []

        if (len(detections) > 0):
            print("--------- Bodies: {} ---------".format(len(detections)))
            for detection in detections:
                x1 = int(detection[0])
                y1 = int(detection[1])
                x2 = int(detection[2])
                y2 = int(detection[3])
                conf = detection[4]
                print("{:.1f} ({},{})-({},{})".format(conf, x1, y1, x2, y2))
                persons.append([x1, y1, x2, y2])
                self._conf.append(float(conf))

        return persons

    def _get_detection(self, infer_result):

        detections = []
        height, width = self.image.shape[:2]
        for r in infer_result[0][0]:
            conf = r[2]
            if (conf > self.threshold):
                x1 = int(r[3] * width)
                y1 = int(r[4] * height)
                x2 = int(r[5] * width)
                y2 = int(r[6] * height)
                detections.append([x1, y1, x2, y2, conf])

        return detections

    @property
    def conf(self):
        return self._conf
