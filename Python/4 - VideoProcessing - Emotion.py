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
from ClassesML.FaceEmotionDetectionModel import *
from Classes.Hyperparameters import *

from ClassesML.Tracker import *
from Classes.JSONUtilities import *
from enum import Enum
from typing import Dict
class DetectionType(Enum):

    EMOTION_OV = 1

class Emotion:

    def __init__(self, data_frame_dict, path_model, detection_type: DetectionType):

        self.data_frame_dict = data_frame_dict
        self.n_frame = len(self.data_frame_dict.keys())

        self.path_model = path_model
        self.detection_type = detection_type

        if self.detection_type == DetectionType.EMOTION_OV:

            self.box_detection_name = "emotion_ov"
            self.json_name = "Face_Emotions"
            self.color = Drawing.color_red
            self.text_pos = 2
            self.text_model_name_position = (10, 100)

            self.label = ('neutral', 'happy', 'sad', 'surprise', 'anger')

        # Load IDs and box coordinates for persons
        results = JSONUtilities.load_json_as_dictionary(folder_path=path_info.results_folder, filename="Face_IDs")
        results = JSONUtilities.list_dict_to_dict_list(dict_list=results)

        self.persons_results = results["Box"]
        self.ids_results = results["IDs"]

        self._get_model()
        self._set_results_dict()

    def _get_model(self):

        self.face_emotion_recognition = FaceEmotionRecognition(path_model=self.path_model)

    def _set_results_dict(self):

        # For storing face emotion recognition results
        self.emotion_results = {}
        self.emotion_name_results = {}
        self.emotion_prob_results = {}

    def detect_face_emotions(self):

        self.results = {}  # Results to be saved as json

        for i in range(self.n_frame):

            image = copy.deepcopy(self.data_frame_dict[i])

            self.emotion_results[i] = self.face_emotion_recognition.infer(image=image,
                                                                          faces=self.persons_results[i])
            emotion_names = []
            emotion_prob = []
            for j in range(len(self.emotion_results[i])):
                temp = self.label[np.argmax(self.emotion_results[i][j].squeeze())]
                emotion_names.append(temp)
                prob = np.max(self.emotion_results[i][j].squeeze())
                emotion_prob.append(prob)

            self.emotion_name_results[i] = emotion_names
            self.emotion_prob_results[i] = emotion_prob

            self.results[i] = {}
            self.results[i]["IDs"] = self.ids_results[i]
            self.results[i]["Box"] = self.persons_results[i]
            self.results[i]["Emotions"] = self.emotion_name_results[i]
            self.results[i]["Emotions_prob"] = self.emotion_prob_results[i]

            print("Processing face emotion recognition: " + str(i) + "/" + str(self.n_frame))

    def save_video_box_with_emotions(self, folder_path, video_name):

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
            DrawingUtilities.draw_box_with_emotion(image=image, persons=persons_result,
                                                   emotions=self.emotion_name_results[i],
                                                   probs=self.emotion_prob_results[i],
                                                   color=self.color)
            image = cv2.putText(image, "Frame: " + str(i), (10, 50), cv2.FONT_HERSHEY_PLAIN, SCALE_TEXT, Drawing.color_blue, SCALE_TEXT, cv2.LINE_AA)
            self.image_to_save_list.append(image)

        # Save video
        save_path = folder_path + video_name + "_" + self.box_detection_name + "_emotion.mp4"
        VideoUtilities.save_images_to_video(images=self.image_to_save_list, save_path=save_path, image_width=image_width, image_height=image_height)

    def save(self, folder_path: str = ""):

        # Function used to save the box and ids detected in the video
        JSONUtilities.save_dictionary_as_json(data_dict=self.results, folder_path=folder_path, filename=self.json_name)

path_info = PathInfo()

# Video link: https://www.youtube.com/watch?v=QRZcZgSgSHI (Video_test_0)
# Video link: https://www.youtube.com/watch?v=QRZcZgSgSHI (Video_test_1) Different moment in the video
# Video link: https://www.youtube.com/watch?v=E1UdvSojKjM (Video_test_2)

video_names = ["Video_test_0.mp4", "Video_test_1.mp4", "Video_test_2.mp4"]

video_name = video_names[0]

# Extract video first
data_frame_dicts = {}
n_frame = 100

for video_name in video_names:

    video = ProcessVideo(video_path=path_info.path_data_test + video_name)
    data_frame_dict = video.read(n_frame=n_frame)
    data_frame_dicts[video_name] = data_frame_dict

detection_type = DetectionType.EMOTION_OV

path_info.set_results_folder(name=Path(video_name).stem)

# Get video as dictionary of image
video = ProcessVideo(video_path=path_info.path_data_test + video_name)
data_frame_dict = video.read(n_frame=n_frame)

emotion = Emotion(data_frame_dict=data_frame_dict, path_model=path_info.path_model, detection_type=detection_type)
emotion.detect_face_emotions()

emotion.save_video_box_with_emotions(folder_path=path_info.results_folder, video_name=video.video_name)
emotion.save(folder_path=path_info.results_folder)
self = emotion
