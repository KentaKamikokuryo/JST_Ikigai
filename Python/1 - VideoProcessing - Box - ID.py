import copy
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from Classes.PathInfo import PathInfo
from Classes.Video import *
import cv2
import random

from ClassesML.ReIdentification import *
from ClassesML.BodyDetectionModel import *
from ClassesML.FaceDetectionModel import *

from ClassesML.Tracker import *
from Classes.JSONUtilities import *
from enum import Enum
from typing import Dict
from Classes.Hyperparameters import *

hyper_parameters_generator = HyperParametersGenerator(detection_type=DetectionType.BODY_BOX_OV)
self = hyper_parameters_generator

class BoxDetectionReIdentification:

    def __init__(self, data_frame_dict, path_model, hyper_parameters: HyperParameters, is_search: bool = True):

        # Do not use opencv in this class
        self.data_frame_dict = data_frame_dict
        self.path_model = path_model
        self.hyper_parameters = hyper_parameters

        self.detection_type = self.hyper_parameters.detection_type

        # Get path
        if is_search:
            if hyper_parameters.detection_type == DetectionType.BODY_BOX_YOLO:
                self.path_database_id = path_info.path_database_id_search
            if hyper_parameters.detection_type == DetectionType.BODY_BOX_OV:
                self.path_database_id = path_info.path_database_id_search
            if hyper_parameters.detection_type == DetectionType.FACE_BOX_OV:
                self.path_database_id = path_info.path_database_face_id_search
            if hyper_parameters.detection_type == DetectionType.FACE_BOX_MP:
                self.path_database_id = path_info.path_database_face_id_search
            if hyper_parameters.detection_type == DetectionType.FACE_BOX_RF:
                self.path_database_id = path_info.path_database_face_id_search
        else:
            if hyper_parameters.detection_type == DetectionType.BODY_BOX_YOLO:
                self.path_database_id = path_info.path_database_id
            if hyper_parameters.detection_type == DetectionType.BODY_BOX_OV:
                self.path_database_id = path_info.path_database_id
            if hyper_parameters.detection_type == DetectionType.FACE_BOX_OV:
                self.path_database_id = path_info.path_database_face_id
            if hyper_parameters.detection_type == DetectionType.FACE_BOX_MP:
                self.path_database_id = path_info.path_database_face_id
            if hyper_parameters.detection_type == DetectionType.FACE_BOX_RF:
                self.path_database_id = path_info.path_database_face_id

        if self.detection_type == DetectionType.BODY_BOX_YOLO:
            self.box_detection_name = "body_box_yolo"
            self.json_name = "IDs"
            self.color = Drawing.color_red
            self.text_pos = 2
            self.text_model_name_position = (10, 100)

        if self.detection_type == DetectionType.BODY_BOX_OV:
            self.box_detection_name = "body_box_ov"
            self.json_name = "IDs"
            self.color = Drawing.color_blue
            self.text_pos = 4
            self.text_model_name_position = (10, 150)

        if self.detection_type == DetectionType.FACE_BOX_OV:
            self.box_detection_name = "face_box_ov"
            self.json_name = "Face_IDs"
            self.color = Drawing.color_red
            self.text_pos = 2
            self.text_model_name_position = (10, 100)

        if self.detection_type == DetectionType.FACE_BOX_MP:
            self.box_detection_name = "face_box_mp"
            self.json_name = "Face_IDs"
            self.color = Drawing.color_blue
            self.text_pos = 4
            self.text_model_name_position = (10, 150)

        if self.detection_type == DetectionType.FACE_BOX_RF:
            self.box_detection_name = "face_box_rf"
            self.json_name = "Face_IDs"
            self.color = Drawing.color_blue
            self.text_pos = 6
            self.text_model_name_position = (10, 150)

        self.n_frame = len(self.data_frame_dict.keys())

        self._get_models()
        self._set_results_dict()

        self.is_box_detection_done = False
        self.is_ids_detection_done = False

    def _get_models(self):

        if self.detection_type == DetectionType.BODY_BOX_YOLO:

            self.person_detector = PersonBodyDetectionYOLO(self.path_model, threshold=self.hyper_parameters.confidence_threshold)
            self.person_re_identification = PersonBodyReidentification(path_model=self.path_model)
            self.tracker = Tracker()  # Track is still used here just for comparison with new DatabaseIDs class. Will be deleted later

        if self.detection_type == DetectionType.BODY_BOX_OV:

            self.person_detector = PersonBodyDetectorOV(path_model=self.path_model, threshold=self.hyper_parameters.confidence_threshold)
            self.person_re_identification = PersonBodyReidentification(path_model=self.path_model)
            self.tracker = Tracker()

        if self.detection_type == DetectionType.FACE_BOX_OV:

            self.person_detector = PersonFaceDetectorOV(path_model=self.path_model, threshold=self.hyper_parameters.confidence_threshold)
            self.person_re_identification = PersonFaceReidentification(path_model=self.path_model)
            self.tracker = Tracker()

        if self.detection_type == DetectionType.FACE_BOX_MP:

            self.person_detector = PersonFaceDetectorMP(path_model=self.path_model, threshold=self.hyper_parameters.confidence_threshold)
            self.person_re_identification = PersonFaceReidentification(path_model=self.path_model)
            self.tracker = Tracker()

        if self.detection_type == DetectionType.FACE_BOX_RF:

            self.person_detector = PersonFaceDetectorRF(path_model=self.path_model, threshold=self.hyper_parameters.confidence_threshold)
            self.person_re_identification = PersonFaceReidentification(path_model=self.path_model)
            self.tracker = Tracker()

        self.database_ids = DatabaseIDs(path_database_id=self.path_database_id)

    def _set_results_dict(self):

        self.detections_results = {}
        self.persons_results = {}
        self.conf_results = {}

    def detect_box(self):

        for i in range(self.n_frame):

            image = copy.deepcopy(self.data_frame_dict[i])
            self.detections_results[i], self.persons_results[i] = self.person_detector.infer(image=image)
            self.conf_results[i] = self.person_detector.conf

            print("Processing body box detection at frame: " + str(i) + "/" + str(self.n_frame))

    def detect_ids(self):

        self.identifies_results = {}  # Neural network output for person identification
        self.ids_results = {}  # IDs for this video
        self.image_boxes = {}  # Coordinates of box representing each person
        self.ids_db_results = {}   # IDs for database of person
        self.results = {}  # Results saved and used for mediapipe on each person

        for i in range(self.n_frame):

            print("")
            print("Reidentification at frame: " + str(i))

            image = copy.deepcopy(self.data_frame_dict[i])
            persons_result = copy.deepcopy(self.persons_results[i])

            # Do identification
            self.identifies_results[i], self.image_boxes[i] = self.person_re_identification.infer(image=image, persons=persons_result)

            self.ids_results[i] = self.tracker.getIds(identifies=self.identifies_results[i], persons=persons_result)
            self.ids_db_results[i] = self.database_ids.get_ids(identifies_new=self.identifies_results[i],
                                                               persons=persons_result,
                                                               image_boxes=self.image_boxes[i],
                                                               DETECTION_THRESHOLD_UP=self.hyper_parameters.DETECTION_THRESHOLD_UP,
                                                               DETECTION_THRESHOLD_DOWN=self.hyper_parameters.DETECTION_THRESHOLD_DOWN,
                                                               DISTANCE_THRESHOLD=self.hyper_parameters.DISTANCE_THRESHOLD,
                                                               DETECTION_N=self.hyper_parameters.DETECTION_N)

            self.results[i] = {}
            self.results[i]["IDs"] = self.ids_db_results[i]
            self.results[i]["Box"] = persons_result

    def save_video_box(self, folder_path, video_name):

        self.image_to_save_list = []

        SCALE_TEXT = 3

        for i in range(self.n_frame):

            image = copy.deepcopy(self.data_frame_dict[i])
            image_height, image_width, _ = image.shape

            if self.persons_results[i] is not None:
                DrawingUtilities.draw_persons_box(image=image, persons=self.persons_results[i], color=Drawing.color_blue)
                image = cv2.putText(image, "Frame: " + str(i), (10, 50), cv2.FONT_HERSHEY_PLAIN, SCALE_TEXT, Drawing.color_blue, SCALE_TEXT, cv2.LINE_AA)
                image = cv2.putText(image, "YOLO", (10, 100), cv2.FONT_HERSHEY_PLAIN, SCALE_TEXT, Drawing.color_blue, SCALE_TEXT, cv2.LINE_AA)

            self.image_to_save_list.append(image)

        # Save video
        save_path = folder_path + video_name + "_" + self.box_detection_name + ".mp4"
        VideoUtilities.save_images_to_video(images=self.image_to_save_list, save_path=save_path, image_width=image_width, image_height=image_height)

    def save_video_box_with_ids(self, folder_path, video_name):

        self.image_to_save_list = []

        TRACKING_MAX = 50
        SCALE_TEXT = 3

        # Use random colors
        colors = []
        for i in range(TRACKING_MAX):
            b = random.randint(0, 255)
            g = random.randint(0, 255)
            r = random.randint(0, 255)
            colors.append((b, g, r))

        for i in range(self.n_frame):

            image = copy.deepcopy(self.data_frame_dict[i])
            image_height, image_width, _ = image.shape

            persons_result = copy.deepcopy(self.persons_results[i])
            DrawingUtilities.draw_box_with_ids(image=image, persons=persons_result, ids_results=self.ids_db_results[i], colors=colors)
            image = cv2.putText(image, "Frame: " + str(i), (10, 50), cv2.FONT_HERSHEY_PLAIN, SCALE_TEXT, Drawing.color_blue, SCALE_TEXT, cv2.LINE_AA)
            self.image_to_save_list.append(image)

        # Save video
        save_path = folder_path + video_name + "_" + self.box_detection_name + "_ids.mp4"
        VideoUtilities.save_images_to_video(images=self.image_to_save_list, save_path=save_path, image_width=image_width, image_height=image_height)

    def save(self, folder_path: str = ""):

        # Function used to save the box and ids detected in the video
        JSONUtilities.save_dictionary_as_json(data_dict=self.results, folder_path=folder_path, filename=self.json_name)

class ComparisonManager():

    def __init__(self, data_frame_dict: dict):

        self.data_frame_dict = data_frame_dict
        self.n_frame = len(self.data_frame_dict.keys())

        self._image_list = []

    def run(self, box_detection_dict: Dict[str, BoxDetectionReIdentification], folder_path, video_name):

        self.box_detection_dict = box_detection_dict

        self._image_list = []
        image_height = 1
        image_width = 1
        SCALE_TEXT = 3

        video_name_suffix = ""

        for box_detection in self.box_detection_dict.values():
            video_name_suffix += box_detection.box_detection_name
            video_name_suffix += "_"

        video_name_suffix = video_name_suffix[:-1]

        for i in range(self.n_frame):

            image = copy.deepcopy(self.data_frame_dict[i])
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_height, image_width, _ = image.shape

            image = cv2.putText(image, "Frame: " + str(i), (10, 50), cv2.FONT_HERSHEY_PLAIN, SCALE_TEXT,
                                Drawing.color_blue, SCALE_TEXT, cv2.LINE_AA)

            # Loop through models
            for box_detection in self.box_detection_dict.values():

                image = cv2.putText(image, box_detection.box_detection_name, box_detection.text_model_name_position,
                                    cv2.FONT_HERSHEY_PLAIN, SCALE_TEXT, box_detection.color, SCALE_TEXT, cv2.LINE_AA)

                image = DrawingUtilities.draw_box_with_confidence(image=image,
                                                                  persons=box_detection.persons_results[i],
                                                                  confs=box_detection.conf_results[i],
                                                                  color=box_detection.color,
                                                                  text_pos=box_detection.text_pos)

            self._image_list.append(image)
            print("Drawing 2D box at frame: " + str(i) + "/" + str(self.n_frame))

        save_path = folder_path + video_name + "_" + video_name_suffix + ".mp4"
        VideoUtilities.save_images_to_video(images=self._image_list,
                                            save_path=save_path,
                                            image_height=image_height,
                                            image_width=image_width)

path_info = PathInfo()

# Video link: https://www.youtube.com/watch?v=QRZcZgSgSHI (Video_test_0)
# Video link: https://www.youtube.com/watch?v=QRZcZgSgSHI (Video_test_1) Different moment in the video
# Video link: https://www.youtube.com/watch?v=E1UdvSojKjM (Video_test_2)

video_names = ["Video_test_0.mp4", "Video_test_1.mp4", "Video_test_2.mp4"]

detection_types = [DetectionType.BODY_BOX_OV, DetectionType.BODY_BOX_YOLO]
detection_types = [DetectionType.FACE_BOX_MP, DetectionType.FACE_BOX_OV]

detection_type = detection_types[0]

video_name = video_names[0]
n_frame = 10

######################
# Extract video data #
######################
data_frame_dicts = {}

for video_name in video_names:
    # Get video as dictionary of image
    video = ProcessVideo(video_path=path_info.path_data_test + video_name)
    data_frame_dict = video.read(n_frame=n_frame)

    data_frame_dicts[video_name] = data_frame_dict

####################################################################################
# Hyperparameters search to find the best combination of box and ID identification #
####################################################################################
is_search = False
is_comparison = False
is_final = True

if is_search:

    count = 0
    scores = []
    path_info = PathInfo()

    # Generate all hyperparameters
    hyper_parameters_generator = HyperParametersGenerator(detection_type=DetectionType.BODY_BOX_OV)
    hyper_combination_dict = hyper_parameters_generator.generate()

    # Clean previous search
    if (os.path.exists(path_info.search_folder)):
        shutil.rmtree(path=path_info.search_folder)

    for count in range(len(hyper_combination_dict)):

        hyper_parameters = HyperParameters(hyper_dict=hyper_combination_dict[count])

        for video_name in video_names:

            data_frame_dict = data_frame_dicts[video_name]

            path_info.set_results_folder_hyper(count=count, name=Path(video_name).stem)

            # Box and id detection
            box_detection_re_id = BoxDetectionReIdentification(data_frame_dict=data_frame_dict,
                                                               path_model=path_info.path_model,
                                                               hyper_parameters=hyper_parameters,
                                                               is_search=is_search)
            box_detection_re_id.detect_box()
            box_detection_re_id.detect_ids()

            box_detection_re_id.save(folder_path=path_info.results_folder)
            box_detection_re_id.save_video_box(folder_path=path_info.results_folder, video_name=Path(video_name).stem)
            box_detection_re_id.save_video_box_with_ids(folder_path=path_info.results_folder, video_name=Path(video_name).stem)

        # Get number of person detected (Compute score based on real number of persons
        if os.path.exists(box_detection_re_id.path_database_id):
            files = os.listdir(box_detection_re_id.path_database_id)
        else:
            files = []

        n_ids_detected = len(files) // 2
        n_ids_true = 5

        scores.append(n_ids_detected / n_ids_true)

        # Save results
        count += 1

    hyper_parameters_generator.set_scores(scores=scores)
    hyper_parameters_generator.save(path=path_info.search_folder)

#####################################################
# Make comparison video for body and face detection #
#####################################################
if is_comparison:

    hyper_combination_best_dict = HyperParametersGenerator.get_best_hyper_each_detection_type()

    path_info = PathInfo()

    for video_name in video_names:

        data_frame_dict = data_frame_dicts[video_name]

        # Prepare folder for results. The folder will have the video name
        path_info.set_results_folder(name=Path(video_name).stem)

        # Dictionary that hold each class for each type of detection
        box_detection_re_id_body_dict = {}
        box_detection_re_id_face_dict = {}

        for detection_type in hyper_combination_best_dict.keys():

            hyper_parameters = HyperParameters(hyper_dict=hyper_combination_best_dict[detection_type])

            # Box and id detection
            box_detection_re_id = BoxDetectionReIdentification(data_frame_dict=data_frame_dict,
                                                               path_model=path_info.path_model,
                                                               hyper_parameters=hyper_parameters,
                                                               is_search=False)
            box_detection_re_id.detect_box()

            if detection_type in [DetectionType.BODY_BOX_OV, DetectionType.BODY_BOX_YOLO]:
                box_detection_re_id_body_dict[detection_type.name] = box_detection_re_id
            else:
                box_detection_re_id_face_dict[detection_type.name] = box_detection_re_id

        # Create video that show two box detection
        comparison_manager = ComparisonManager(data_frame_dict=data_frame_dict)
        comparison_manager.run(box_detection_dict=box_detection_re_id_body_dict, folder_path=path_info.results_folder,
                               video_name=Path(video_name).stem)

        comparison_manager = ComparisonManager(data_frame_dict=data_frame_dict)
        comparison_manager.run(box_detection_dict=box_detection_re_id_face_dict, folder_path=path_info.results_folder,
                               video_name=Path(video_name).stem)

#########################################
# Only use best model for face and body #
#########################################
if is_final:

    hyper_combination_best_dict = HyperParametersGenerator.get_best_hyper_each_detection_type()

    for detection_type in hyper_combination_best_dict.keys():

        hyper_parameters = HyperParameters(hyper_dict=hyper_combination_best_dict[detection_type])

        for video_name in video_names:

            data_frame_dict = data_frame_dicts[video_name]

            path_info.set_results_folder(name=Path(video_name).stem)

            # Box and id detection
            box_detection_re_id = BoxDetectionReIdentification(data_frame_dict=data_frame_dict,
                                                               path_model=path_info.path_model,
                                                               hyper_parameters=hyper_parameters,
                                                               is_search=False)
            box_detection_re_id.detect_box()
            box_detection_re_id.detect_ids()

            self = box_detection_re_id

            box_detection_re_id.save(folder_path=path_info.results_folder)
            box_detection_re_id.save_video_box(folder_path=path_info.results_folder, video_name=Path(video_name).stem)
            box_detection_re_id.save_video_box_with_ids(folder_path=path_info.results_folder, video_name=Path(video_name).stem)
