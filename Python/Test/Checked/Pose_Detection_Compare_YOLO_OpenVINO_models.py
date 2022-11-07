import cv2
from Classes.PathInfo import PathInfo
from Classes.Video import ProcessVideo, VideoUtilities
from Classes.Drawing import Drawing, DrawingUtilities
from ClassesML.ModelMLUtilities import *
from Kenta.TempClasses.PathInfo import KentaPathInfo
import copy

class Model:

    def __init__(self, path_model, model_name):

        self.device = "CPU"
        self.num_requests = 2
        self.threshold = 0.5

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

class PersonBodyDetectionYOLO:

    def __init__(self, path_model:  str):

        self.model_name = "yolov5s"

        # Download pytorch model
        self.model = ModelMLUtilities.load_online_pytorch_model(url='ultralytics/yolov5', path_model=path_model + self.model_name)

        # since we are only interested in detecting person
        self.model.classes = [0]

        # set model parameter for confidence limit
        """
        conf = 0.25 - NMS confidence threshold
        iou = 0.45 - NMS IoU threshold
        agnostic = False - NMS class-agnostic
        multi_label = False - NMS multiple labels per box
        classes = None - (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
        max_det = 1000 - maximum number of detections per image
        amp = False - Automatic Mixed Precision (AMP) inference
        """

        self.model.conf = 0.5
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

        # Loop through all crops from YOLO
        for n in range(len(crops)):
            crop = crops[n]
            if "person" in crop["label"]:
                box = [int(crop["box"][0]), int(crop["box"][1]), int(crop["box"][2]), int(crop["box"][3])]
                persons.append(box)
                self._conf.append(float(crop["conf"]))

        return persons

    def _get_detection(self):

        yolo_result = self.model(self.image)
        yolo_result.print()

        return yolo_result

    @property
    def conf(self):
        return self._conf

class PersonBodyDetector(Model):

    def __init__(self, path_model):

        self.model_name = "person-detection-retail-0013"

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
            print("-------------------")
            for detection in detections:
                x1 = int(detection[0])
                y1 = int(detection[1])
                x2 = int(detection[2])
                y2 = int(detection[3])
                conf = detection[4]
                print("{:.1f} ({},{})-({},{})".format(conf, x1, y1, x2, y2))
                persons.append([x1, y1, x2, y2])
                self._conf.append(conf)

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

class Manager():

    def __init__(self, data_frame_dict, path_model):

        self.data_frame_dict = data_frame_dict
        self.path_model = path_model

        self.n_frame = len(self.data_frame_dict.keys())

        self._get_models()
        self._set_results_dict()

    def process(self):

        print("Start to process for detecting person")
        self._detect_person_YOLO()

        print("Start to process for detecting person ids")
        self._detect_person_OV()

    def _get_models(self):

        # Get models for human detection (box)
        self.person_detector_YOLO = PersonBodyDetectionYOLO(self.path_model)

        self.person_detector_OV = PersonBodyDetector(path_model=self.path_model)

    def _set_results_dict(self):

        # YOLO
        self._persons_results_YOLO = {}
        self._detection_results_YOLO = {}
        self._conf_results_YOLO = {}

        # OpenVINO
        self._persons_results_OV = {}
        self._detection_results_OV = {}
        self._conf_results_OV = {}

    def _detect_person_YOLO(self):

        for i in range(self.n_frame):

            image = copy.deepcopy(self.data_frame_dict[i])

            self._detection_results_YOLO[i], self._persons_results_YOLO[i] = self.person_detector_YOLO.infer(image=image)
            self._conf_results_YOLO[i] = self.person_detector_YOLO.conf
            print("Processing box detection with YOLO at frame: " + str(i) + "/" + str(self.n_frame))

    def _detect_person_OV(self):

        for i in range(self.n_frame):

            image = copy.deepcopy(self.data_frame_dict[i])

            self._detection_results_OV[i], self._persons_results_OV[i] = self.person_detector_OV.infer(image=image)
            self._conf_results_OV[i] = self.person_detector_OV.conf
            print("Processing box detection with OpenVINO at frame: " + str(i) + "/" + str(self.n_frame))

    @property
    def person_results_YOLO(self):
        return self._persons_results_YOLO

    @property
    def detection_results_YOLO(self):
        return self._detection_results_YOLO

    @property
    def conf_results_YOLO(self):
        return self._conf_results_YOLO

    @property
    def person_results_OV(self):
        return self._persons_results_OV

    @property
    def detection_results_OV(self):
        return self._detection_results_OV

    @property
    def conf_results_OV(self):
        return self._conf_results_OV

class VideoManager():

    def __init__(self,data_frame_dict: dict, plot_parameters: dict):

        self.SCALE_IMAGE = plot_parameters["scale_image"]
        self.SCALE_ID = plot_parameters["scale_id"]
        self.TRACKING_MAX = plot_parameters["tracking_max"]
        self.MARGIN = plot_parameters["margin"]
        self.SCALE_TEXT = plot_parameters["scale_text"]
        self.THICKNESS_TEXT = plot_parameters["thickness_text"]

        self.data_frame_dict = data_frame_dict
        self.n_frame = len(self.data_frame_dict.keys())

        self._image_list = []

    def write_bbox_comparison(self,
                              person_results_yolo: dict,
                              person_results_ov: dict,
                              conf_yolo: dict,
                              conf_ov: dict,
                              video_save_folder,
                              video_name):

        self._image_list = []
        image_height = 1
        image_width = 1

        for i in range(self.n_frame):

            image = copy.deepcopy(self.data_frame_dict[i])
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_height, image_width, _ = image.shape

            # For drawing YOLO information
            image = DrawingUtilities.put_highlight_text(image=image,
                                                        text="box detection model: YOLO",
                                                        org=(0, 30),
                                                        scale_text=self.SCALE_TEXT + .5,
                                                        color=Drawing.color_red,
                                                        thickness=self.THICKNESS_TEXT)

            # For drawing Open VINO information
            image = DrawingUtilities.put_highlight_text(image=image,
                                                        text="box detection model: OpenVINO",
                                                        org=(0, 60),
                                                        scale_text=self.SCALE_TEXT + .5,
                                                        color=Drawing.color_blue,
                                                        thickness=self.THICKNESS_TEXT)

            # YOLO
            image = self._draw_person_box(image=image,
                                          persons=person_results_yolo[i],
                                          confs=conf_yolo[i],
                                          color=Drawing.color_red,
                                          text_pos=2)

            # OpenVINO
            image = self._draw_person_box(image=image,
                                          persons=person_results_ov[i],
                                          confs=conf_ov[i],
                                          color=Drawing.color_blue,
                                          text_pos=4)

            self._image_list.append(image)
            print("Drawing 2D box at frame: " + str(i) + "/" + str(self.n_frame))

        save_path = video_save_folder + "_" + video_name + "_BBox_Comparison.mp4"
        VideoUtilities.save_images_to_video(images=self._image_list,
                                            save_path=save_path,
                                            image_height=image_height,
                                            image_width=image_width)

    def _draw_person_box(self, image, persons, confs, color, text_pos: int = 2):

        for j, person in enumerate(persons):

            start_point = (person[0] - self.MARGIN, person[1] - self.MARGIN)
            end_point = (person[2] + self.MARGIN, person[3] + self.MARGIN)

            image = DrawingUtilities.put_highlight_text(image=image,
                                                        text="conf: {:.2f}".format(confs[j]),
                                                        org=(person[0] - self.MARGIN, person[1] - self.MARGIN * text_pos),
                                                        scale_text=self.SCALE_TEXT,
                                                        color=color,
                                                        thickness=self.THICKNESS_TEXT)

            image = DrawingUtilities.put_highlight_rectangle(image=image,
                                                             pt1=start_point,
                                                             pt2=end_point,
                                                             color=color,
                                                             thickness=self.THICKNESS_TEXT)

        return image

plot_parameters = {"scale_image": 1,
                   "scale_id": 0.2,
                   "scale_text": 2,
                   "thickness_text": 1,
                   "tracking_max": 50,
                   "margin": 10}

pathInfo = PathInfo()
video_folder = pathInfo.path_data_test
video_name = "Experimental_Video_2"
video_format = ".mp4"

cap = cv2.VideoCapture(video_folder + video_name + video_format)
processVideo = ProcessVideo(video_path=video_folder + video_name + video_format)
data_frame_dict = processVideo.read(n_frame=200)

holistic_detection_parameters = dict(static_image_mode=False,
                                     min_detection_confidence=0.2,
                                     model_complexity=1,
                                     min_tracking_confidence=0.2,
                                     smooth_landmarks=True)

manager = Manager(data_frame_dict=data_frame_dict,
                  path_model=pathInfo.path_model)
manager.process()

videoManager = VideoManager(data_frame_dict=data_frame_dict, plot_parameters=plot_parameters)
videoManager.write_bbox_comparison(person_results_yolo=manager.person_results_YOLO,
                                   person_results_ov=manager.person_results_OV,
                                   conf_yolo=manager.conf_results_YOLO,
                                   conf_ov=manager.conf_results_OV,
                                   video_save_folder=KentaPathInfo().saved_video_path,
                                   video_name=video_name)

