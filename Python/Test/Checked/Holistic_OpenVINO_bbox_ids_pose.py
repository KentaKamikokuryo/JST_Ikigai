import numpy as np
import random
import cv2
import mediapipe as mp
from Classes.PathInfo import PathInfo
from Classes.Video import ProcessVideo, VideoUtilities
from Classes.Drawing import Drawing, DrawingUtilities
from ClassesML.ModelMLUtilities import *
from ClassesML.Tracker import Tracker
from Kenta.TempClasses.PathInfo import KentaPathInfo
from ClassesML.BodyDetectionModel import PersonBodyDetectionYOLO, PersonBodyDetectorOV
import copy

class Model:

    def __init__(self, path_model, model_name):

        self.device = "CPU"
        self.num_requests = 2
        self.threshold = 0.4

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

# class PersonBodyDetectionYOLO:
#
#     def __init__(self, path_model:  str):
#
#         self.model_name = "yolov5s"
#
#         # Download pytorch model
#         self.model = ModelMLUtilities.load_online_pytorch_model(url='ultralytics/yolov5', path_model=path_model + self.model_name)
#
#         # since we are only interested in detecting person
#         self.model.classes = [0]
#
#     def infer(self, image):
#
#         self.image = copy.deepcopy(image)
#
#         self.image.flags.writeable = False
#
#         yolo_result = self._get_detection()
#
#         self.image.flags.writeable = True
#
#         persons = self._get_persons(results=yolo_result)
#
#         return yolo_result, persons
#
#     def _get_persons(self, results):
#
#         # Get crops
#         crops = results.crop(save=False)  # cropped detections dictionary
#
#         persons = []
#
#         # Loop through all crops from YOLO
#         for n in range(len(crops)):
#             crop = crops[n]
#             if "person" in crop["label"]:
#                 box = [int(crop["box"][0]), int(crop["box"][1]), int(crop["box"][2]), int(crop["box"][3])]
#                 persons.append(box)
#
#         return persons
#
#     def _get_detection(self):
#
#         yolo_result = self.model(self.image)
#         yolo_result.print()
#
#         return yolo_result

# class PersonBodyDetector(Model):
#
#     def __init__(self, path_model):
#
#         self.model_name = "person-detection-retail-0013"
#         self._conf = []
#
#         super().__init__(path_model, self.model_name)
#
#     def infer(self, image):
#
#         self.image = image
#         self.scaled_frame, self.scale_h, self.scale_w = self.scale_frame(image)
#
#         infer_result = self.compiled_model([self.scaled_frame])[self.output_layer]
#
#         detections = self._get_detection(infer_result)
#         persons = self._get_persons(detections)
#
#         return detections, persons
#
#     def _get_persons(self, detections):
#
#         persons = []
#
#         if (len(detections) > 0):
#             print("-------------------")
#             for detection in detections:
#                 x1 = int(detection[0])
#                 y1 = int(detection[1])
#                 x2 = int(detection[2])
#                 y2 = int(detection[3])
#                 conf = detection[4]
#                 print("{:.1f} ({},{})-({},{})".format(conf, x1, y1, x2, y2))
#                 persons.append([x1, y1, x2, y2])
#                 self._conf.append(conf)
#
#         return persons
#
#     def _get_detection(self, infer_result):
#
#         detections = []
#         height, width = self.image.shape[:2]
#         for r in infer_result[0][0]:
#             conf = r[2]
#             if (conf > self.threshold):
#                 x1 = int(r[3] * width)
#                 y1 = int(r[4] * height)
#                 x2 = int(r[5] * width)
#                 y2 = int(r[6] * height)
#                 detections.append([x1, y1, x2, y2, conf])
#
#         return detections
#
#     @property
#     def conf(self):
#         return self._conf

class PersonBodyReidentification(Model):

    def __init__(self, path_model):

        self.model_name = "person-reidentification-retail-0288"

        super().__init__(path_model, self.model_name)

    def infer(self, image, persons):

        self.image = image

        identifies = np.zeros((len(persons), 256))
        image_boxes = []

        for i, person in enumerate(persons):

            # Acquisition of each person's image
            img = image[person[1]: person[3], person[0]: person[2]]
            h, w = img.shape[:2]

            if (h == 0 or w == 0):
                continue

            scaled_img, self.scale_h, self.scale_w = self.scale_frame(img)

            # Identification information acquisition
            results = self.compiled_model([scaled_img])[self.output_layer]
            identifies[i] = results[0, :]

            image_boxes.append(copy.deepcopy(img))

        return identifies, image_boxes

class MediaPipe():

    def __init__(self, holistic_detection_params: dict):

        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_holistic = mp.solutions.holistic
        self._holistic_detection_parameters = holistic_detection_params

    @property
    def mp_drawing(self):
        return self._mp_drawing

    @property
    def mp_holistic(self):
        return self._mp_holistic

    @property
    def holistic_detection_parameters(self):
        return self._holistic_detection_parameters


class Manager():

    def __init__(self, mediaPipe: MediaPipe, data_frame_dict, path_model, box_model: str = "YOLO"):

        self.data_frame_dict = data_frame_dict
        self.path_model = path_model
        self.mediaPipe = mediaPipe
        self.box_model = box_model

        self.n_frame = len(self.data_frame_dict.keys())

        self._get_models()
        self._set_results_dict()

    def process(self):

        print("Start to process for detecting person")
        self._detect_person()

        print("Start to process for detecting person ids")
        self._detect_person_ids()

        # print("Start to process for detecting holistic information")
        # self._detect_person_pose()

    def _get_models(self):

        # Get models for human detection (box)
        if self.box_model == "YOLO":
            self.person_detector = PersonBodyDetectionYOLO(self.path_model)

        elif self.box_model == "OpenVINO":
            self.person_detector = PersonBodyDetectorOV(path_model=self.path_model)

        else:
            self.box_model = "YOLO"
            self.person_detector = PersonBodyDetectionYOLO(self.path_model)


        self.parson_re_identification = PersonBodyReidentification(path_model=self.path_model)
        self.tracker = Tracker()

        self.mp_holistic = self.mediaPipe.mp_holistic

    def _set_results_dict(self):

        # for storing person detection results
        self._persons_results = {}
        self._detection_results = {}
        self._confidence_result = {}

        # for storing person re-identification results
        self._identifies_results = {}
        self._ids_results = {}

        # for storing person pose results
        self._results_mp = {}
        self._results_w = {}
        self._results_i = {}
        self._results_v = {}

    def _detect_person(self):

        for i in range(self.n_frame):

            image = copy.deepcopy(self.data_frame_dict[i])

            self._detection_results[i], self._persons_results[i] = self.person_detector.infer(image=image)
            self._confidence_result[i] = self.person_detector.conf

            print("Processing box detection at frame: " + str(i) + "/" + str(self.n_frame))

    def _detect_person_ids(self):

        for i in range(self.n_frame):

            image = copy.deepcopy(self.data_frame_dict[i])

            self._identifies_results[i], _ = self.parson_re_identification.infer(image=image,
                                                                              persons=self._persons_results[i])

            self._ids_results[i] = self.tracker.getIds(identifies=self._identifies_results[i],
                                                          persons=self._persons_results[i])

            print("Processing ID at frame: " + str(i) + "/" + str(self.n_frame))

    def _detect_person_pose(self):

        MARGIN = 20

        for i in range(self.n_frame):

            image = copy.deepcopy(self.data_frame_dict[i])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            self._results_mp[i] = {}
            self._results_w[i] = {}
            self._results_i[i] = {}
            self._results_v[i] = {}

            for j, person in enumerate(self._persons_results[i]):

                xmin = person[0]
                xmax = person[2]
                ymin = person[1]
                ymax = person[3]

                img = image[(int(ymin) - MARGIN):(int(ymax) + MARGIN), (int(xmin) - MARGIN):(int(xmax) + MARGIN)]
                image_height, image_width, _ = img.shape

                with self.mp_holistic.Holistic(**self.mediaPipe.holistic_detection_parameters) as pose:

                    self._results_mp[i][j] = pose.process(image=img)

                    if self._results_mp[i][j].pose_landmarks.landmark is not None:

                        landmarks = self._results_mp[i][j].pose_landmarks.landmark
                        landmarks_w = self._results_mp[i][j].pose_world_landmarks.landmark
                        n_landmarks = len(landmarks)
                        self._results_i[i][j] = np.array([[p.x * image_width, p.y * image_height] for p in landmarks])
                        self._results_w[i][j] = np.array([[-landmarks_w[k].z, landmarks_w[k].x, -landmarks_w[k].y] for k in range(n_landmarks)])
                        self._results_v[i][j] = np.array([[landmarks_w[k].visibility] for k in range(n_landmarks)])

                    else:

                        self._results_i[i][j] = np.array([])
                        self._results_w[i][j] = np.array([])
                        self._results_v[i][j] = np.array([])

            print("Processing Pose at frame: " + str(i) + "/" + str(self.n_frame))

    @property
    def person_results(self):
        return self._persons_results

    @property
    def detection_results(self):
        return self._detection_results

    @property
    def identifies_results(self):
        return self._identifies_results

    @property
    def confidence_results(self):
        return self._confidence_result

    @property
    def ids_results(self):
        return self._ids_results

    @property
    def results_mp(self):
        return self._results_mp

    @property
    def results_w(self):
        return self._results_w

    @property
    def results_i(self):
        return self._results_i

    @property
    def results_v(self):
        return self._results_v


class VideoManager():

    def __init__(self, mediaPipe: MediaPipe, data_frame_dict: dict, plot_parameters: dict, box_model: str = "YOLO"):

        self.SCALE_IMAGE = plot_parameters["scale_image"]
        self.TRACKING_MAX = plot_parameters["tracking_max"]
        self.MARGIN = plot_parameters["margin"]
        self.SCALE_TEXT = plot_parameters["scale_text"]
        self.THICKNESS_TEXT = plot_parameters["thickness_text"]

        self.data_frame_dict = data_frame_dict
        self.n_frame = len(self.data_frame_dict.keys())
        self.box_model = box_model

        self.mediaPipe = mediaPipe
        self.mp_drawing = self.mediaPipe.mp_drawing
        self.mp_holistic = self.mediaPipe.mp_holistic

        self._image_list = []
        self._set_colors_tracking()
        self._set_box_model_info()

    def _set_box_model_info(self):

        if self.box_model == "YOLO":

            self.box_color = Drawing.color_red

        elif self.box_model == "OpenVINO":

            self.box_color = Drawing.color_blue

        else:

            self.box_model = "YOLO"
            self.box_color = Drawing.color_red

    def generate_video_box(self,
                           person_results: dict,
                           conf: dict,
                           video_save_folder,
                           video_name):

        self._image_list = []
        image_height = 1
        image_width = 1

        for i in range(self.n_frame):

            image = copy.deepcopy(self.data_frame_dict[i])
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_height, image_width, _ = image.shape

            image = self._draw_bbox(image=image, persons=person_results[i], color=Drawing.color_white, conf=conf[i])

            # image = cv2.putText(image, "box detection model: " + str(self.box_model),
            #                     (10, 100),
            #                     cv2.FONT_HERSHEY_PLAIN,
            #                     SCALE_TEXT,
            #                     self.box_color,
            #                     SCALE_TEXT,
            #                     cv2.LINE_AA)

            self._image_list.append(image)
            print("Drawing 2D box at frame: " + str(i) + "/" + str(self.n_frame))

        save_path = video_save_folder + "_" + video_name + "_" + self.box_model + ".mp4"
        VideoUtilities.save_images_to_video(images=self._image_list,
                                            save_path=save_path,
                                            image_height=image_height,
                                            image_width=image_width)

    def generate_video_box_with_ids(self,
                                    person_results: dict,
                                    ids_results: dict,
                                    conf: dict,
                                    video_save_folder, video_name):

        self._image_list = []

        for i in range(self.n_frame):

            image = copy.deepcopy(self.data_frame_dict[i])
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_height, image_width, _ = image.shape

            # image = cv2.putText(image, "box detection model: " + str(self.box_model),
            #                     (10, 100),
            #                     cv2.FONT_HERSHEY_PLAIN,
            #                     SCALE_TEXT,
            #                     self.box_color,
            #                     SCALE_TEXT,
            #                     cv2.LINE_AA)

            image = self._draw_bbox_ids(image=image,
                                        persons=person_results[i],
                                        ids_results=ids_results[i],
                                        conf=conf[i],
                                        colors=self._colors)

            self._image_list.append(image)
            print("Drawing box and id at frame: " + str(i) + "/" + str(self.n_frame))

        save_path = video_save_folder + "_" + video_name + "_" + self.box_model + "_ids.mp4"
        VideoUtilities.save_images_to_video(images=self._image_list,
                                            save_path=save_path,
                                            image_height=image_height,
                                            image_width=image_width)

    def generate_video_box_ids_pose(self,
                                    person_results: dict,
                                    ids_results: dict,
                                    conf: dict,
                                    mp_results: dict,
                                    video_save_folder, video_name):

        self._image_list = []

        for i in range(self.n_frame):

            image = copy.deepcopy(self.data_frame_dict[i])
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_height, image_width, _ = image.shape

            # image = cv2.putText(image, "box detection model: " + str(self.box_model),
            #                     (10, 100),
            #                     cv2.FONT_HERSHEY_PLAIN,
            #                     SCALE_TEXT,
            #                     self.box_color,
            #                     SCALE_TEXT,
            #                     cv2.LINE_AA)

            image = self._draw_bbox_ids_pose(image=image,
                                             persons=person_results[i],
                                             ids_results=ids_results[i],
                                             conf=conf[i],
                                             mp_results=mp_results[i],
                                             colors=self._colors)

            self._image_list.append(image)
            print("Drawing box and id, pose at frame: " + str(i) + "/" + str(self.n_frame))

        save_path = video_save_folder + "_" + video_name + "_" + self.box_model + "_ids_pose.mp4"
        VideoUtilities.save_images_to_video(images=self._image_list,
                                            save_path=save_path,
                                            image_height=image_height,
                                            image_width=image_width)

    def _set_colors_tracking(self):

        self._colors = []

        for i in range(self.TRACKING_MAX):

            b = random.randint(0, 255)
            g = random.randint(0, 255)
            r = random.randint(0, 255)

            self._colors.append((b, g, r))

    def _draw_bbox(self, image, persons, color, conf):

        for j, person in enumerate(persons):

            start_point = (person[0] - self.MARGIN, person[1] - self.MARGIN)
            end_point = (person[2] + self.MARGIN, person[3] + self.MARGIN)

            image = DrawingUtilities.put_highlight_text(image=image,
                                                        text="confidence: {:.2f}".format(conf[j]),
                                                        org=(person[0] - self.MARGIN, person[1] - self.MARGIN * 2),
                                                        scale_text=self.SCALE_TEXT,
                                                        color=color,
                                                        thickness=self.THICKNESS_TEXT)

            image = DrawingUtilities.put_highlight_rectangle(image=image,
                                                             pt1=start_point,
                                                             pt2=end_point,
                                                             color=color,
                                                             thickness=self.THICKNESS_TEXT)

        return image

    def _draw_bbox_ids(self, image, persons, ids_results, conf, colors):

        for i, person in enumerate(persons):

            if (ids_results[i] != -1):

                color = colors[int(ids_results[i])]
                start_point = (person[0] - self.MARGIN, person[1] - self.MARGIN)
                end_point = (person[2] + self.MARGIN, person[3] + self.MARGIN)

                image = DrawingUtilities.put_highlight_text(image=image,
                                                            text="Id: {} - Conf: {:.2f}".format(str(ids_results[i]), conf[i]),
                                                            org=(person[0] - self.MARGIN, person[1] - self.MARGIN * 2),
                                                            scale_text=self.SCALE_TEXT,
                                                            color=color,
                                                            thickness=self.THICKNESS_TEXT)

                image = DrawingUtilities.put_highlight_rectangle(image=image,
                                                                 pt1=start_point,
                                                                 pt2=end_point,
                                                                 color=color,
                                                                 thickness=self.THICKNESS_TEXT)

        return image

    def _draw_bbox_ids_pose(self, image, persons, ids_results, conf, mp_results, colors):

        MARGIN = 10

        for i, person in enumerate(persons):

            if (ids_results[i] != -1):

                color = colors[int(ids_results[i])]
                start_point = (person[0] - self.MARGIN, person[1] - self.MARGIN)
                end_point = (person[2] + self.MARGIN, person[3] + self.MARGIN)

                image = DrawingUtilities.put_highlight_text(image=image,
                                                            text="Id: {} - Conf: {:.2f}".format(str(ids_results[i]), conf[i]),
                                                            org=(person[0] - self.MARGIN, person[1] - self.MARGIN * 2),
                                                            scale_text=self.SCALE_TEXT,
                                                            color=color,
                                                            thickness=self.THICKNESS_TEXT)

                image = DrawingUtilities.put_highlight_rectangle(image=image,
                                                                 pt1=start_point,
                                                                 pt2=end_point,
                                                                 color=color,
                                                                 thickness=self.THICKNESS_TEXT)

                xmin = person[0]
                xmax = person[2]
                ymin = person[1]
                ymax = person[3]

                # # 1. Draw face landmarks
                # self.mp_drawing.draw_landmarks(
                #     image[(int(ymin) - MARGIN):(int(ymax) + MARGIN), (int(xmin) - MARGIN):(int(xmax) + MARGIN)],
                #     mp_results[i].face_landmarks, self.mp_holistic.FACEMESH_CONTOURS,
                #     self.mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                #     self.mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                #     )
                #
                # # 2. Right hand
                # self.mp_drawing.draw_landmarks(
                #     image[(int(ymin) - MARGIN):(int(ymax) + MARGIN), (int(xmin) - MARGIN):(int(xmax) + MARGIN)],
                #     mp_results[i].right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                #     landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=4),
                #     connection_drawing_spec=self.mp_drawing.DrawingSpec(color=Drawing.color_white, thickness=2)
                #     )
                #
                # # 3. Left Hand
                # self.mp_drawing.draw_landmarks(
                #     image[(int(ymin) - MARGIN):(int(ymax) + MARGIN), (int(xmin) - MARGIN):(int(xmax) + MARGIN)],
                #     mp_results[i].left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                #     landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=4),
                #     connection_drawing_spec=self.mp_drawing.DrawingSpec(color=Drawing.color_white, thickness=2)
                #     )

                # 4. Pose Detections
                self.mp_drawing.draw_landmarks(
                    image[(int(ymin) - self.MARGIN):(int(ymax) + self.MARGIN), (int(xmin) - self.MARGIN):(int(xmax) + self,MARGIN)],
                    mp_results[i].pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=4),
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(color=Drawing.color_white, thickness=2)
                    )

        return image

plot_parameters = {"scale_image": 1,
                   "scale_text": 3,
                   "thickness_text": 4,
                   "tracking_max": 50,
                   "margin": 10}

box_model = "OpenVINO"
pathInfo = PathInfo()
video_folder = pathInfo.path_data_test
video_folder = KentaPathInfo().path_data_test
video_name = "Experimental_Video_2"
video_format = ".mp4"

cap = cv2.VideoCapture(video_folder + video_name + video_format)
processVideo = ProcessVideo(video_path=video_folder + video_name + video_format)
data_frame_dict = processVideo.read(n_frame=200, resize_param=3)

holistic_detection_parameters = dict(static_image_mode=False,
                                     min_detection_confidence=0.2,
                                     model_complexity=1,
                                     min_tracking_confidence=0.2,
                                     smooth_landmarks=True)

mediaPipe = MediaPipe(holistic_detection_params=holistic_detection_parameters)

manager = Manager(mediaPipe=mediaPipe,
                  data_frame_dict=data_frame_dict,
                  path_model=pathInfo.path_model,
                  box_model=box_model)
manager.process()

videoManager = VideoManager(mediaPipe=mediaPipe, data_frame_dict=data_frame_dict, plot_parameters=plot_parameters, box_model=box_model)

videoManager.generate_video_box(person_results=manager.person_results,
                                video_save_folder=KentaPathInfo().saved_video_path,
                                video_name=video_name,
                                conf=manager.confidence_results)

videoManager.generate_video_box_with_ids(person_results=manager.person_results,
                                         ids_results=manager.ids_results,
                                         conf=manager.confidence_results,
                                         video_save_folder=KentaPathInfo().saved_video_path,
                                         video_name=video_name)
#
# videoManager.generate_video_box_ids_pose(person_results=manager.person_results,
#                                          ids_results=manager.ids_results,
#                                          mp_results=manager.results_mp,
#                                          video_save_folder=KentaPathInfo().saved_video_path,
#                                          video_name=video_name)