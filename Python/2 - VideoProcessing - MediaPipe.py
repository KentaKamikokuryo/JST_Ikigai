import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from Classes.PathInfo import PathInfo
from Classes.Video import *
import cv2
import json
import random
import numpy as np
from ClassesML.ReIdentification import PersonBodyReidentification, PersonFaceReidentification
from ClassesDB.DatabaseIDs import DatabaseIDs
from Classes.JSONUtilities import JSONUtilities
from Classes.Drawing import DrawingUtilities
from Classes.ProcessWebcam import *
from Classes.Hyperparameters import *

class VideoMediaPipe:

    def __init__(self, data_frame_dict, results, parameters):

        self.data_frame_dict = data_frame_dict
        self.persons_results = results["Box"]
        self.ids_results = results["IDs"]

        self.draw_pose = parameters["draw_pose"]
        self.draw_face = parameters["draw_face"]
        self.draw_3D = parameters["draw_3D"]
        self.face = parameters["face"]
        self.body = parameters["body"]

        self.n_frame = len(self.data_frame_dict.keys())

        self.get_models()

        self.MARGIN = 20

    def get_models(self):

        if self.face:

            self.process_face_frame = ProcessFaceFrame()
            self.process_face_frame.set_parameters()

        if self.body:

            self.process_holistic_frame = ProcessHolisticFrame()
            self.process_holistic_frame.set_parameters()

    def process(self):

        self.results_face_mp = {}  # mp - MediaPipe
        self.results_face_w = {}  # w - 3D world coordinates
        self.results_face_i = {}  # i - 2D image coordinates
        self.results_face_v = {}  # v - Visibility

        self.results_hollistic_mp = {}
        self.results_hollistic_w = {}
        self.results_hollistic_i = {}
        self.results_hollistic_v = {}
        self.results_hollistic_mp_pl = {}  # mp_pl - MediaPipe - pose_landmarks

        for i in range(self.n_frame):

            self.results_face_mp[i] = {}
            self.results_face_w[i] = {}
            self.results_face_i[i] = {}
            self.results_face_v[i] = {}

            self.results_hollistic_mp[i] = {}
            self.results_hollistic_w[i] = {}
            self.results_hollistic_i[i] = {}
            self.results_hollistic_v[i] = {}
            self.results_hollistic_mp_pl[i] = {}  # Pose landmark only

            image = copy.deepcopy(self.data_frame_dict[i])
            self.image_height, self.image_width, _ = image.shape

            # Goes through each person
            persons_results = self.persons_results[i]
            ids_results = self.ids_results[i]

            for j, person in enumerate(persons_results):

                id = ids_results[j]

                if (id != -1):

                    # Acquisition of each person's image
                    img = image[(person[1]-self.MARGIN):(person[3]+self.MARGIN), (person[0]-self.MARGIN):(person[2]+self.MARGIN)]
                    h, w = img.shape[:2]

                    if self.face:

                        self.results_face_i[i][id], \
                        self.results_face_w[i][id], \
                        self.results_face_mp[i][id], \
                        self.results_face_v[i][id] = self.process_face_frame.process(image=img)

                    else:

                        self.results_hollistic_i[i][id], \
                        self.results_hollistic_w[i][id], \
                        self.results_hollistic_mp[i][id], \
                        self.results_hollistic_v[i][id] = self.process_holistic_frame.process(image=img)

                        self.results_hollistic_mp_pl[i][id] = self.results_hollistic_mp[i][id].pose_landmarks

            print("Processing frame " + str(i) + "/" + str(self.n_frame))

    def save_video(self, folder_path, video_name):

        self.image_to_save_list = []

        for i in range(self.n_frame):

            image = copy.deepcopy(self.data_frame_dict[i])
            self.image_height, self.image_width, _ = image.shape

            # Goes through each person
            persons_results = self.persons_results[i]
            ids_results = self.ids_results[i]

            for j, person in enumerate(persons_results):

                id = ids_results[j]

                if (id != -1):

                    # Acquisition of each person's image
                    img = image[(person[1]-self.MARGIN):(person[3]+self.MARGIN), (person[0]-self.MARGIN):(person[2]+self.MARGIN)]
                    h, w = img.shape[:2]

                    if (h == 0 or w == 0):
                        continue

                    if self.face:
                        if self.draw_face:
                            if self.results_face_mp[i][id].multi_face_landmarks is not None:
                                DrawingUtilities.draw_face(image=img, results_mp=self.results_face_mp[i][id],
                                                           mp_face_mesh=self.process_face_frame.mp_face_mesh)
                    if self.body:
                        if self.draw_pose:
                            if self.results_hollistic_mp[i][id].pose_landmarks is not None:
                                # DrawingUtilities.draw_pose(image=img, results_mp=self.results_hollistic_mp[i][id], mp_holistic=self.process_holistic_frame.mp_holistic)
                                DrawingUtilities.draw_pose_pl(image=img, results_mp_pl=self.results_hollistic_mp_pl[i][id], mp_holistic=self.process_holistic_frame.mp_holistic)

            self.image_to_save_list.append(image)
            print("Drawing hollistic pose at frame: " + str(i) + "/" + str(self.n_frame))

        # Save video
        save_path = folder_path + video_name + "_" + "mediapipe" + ".mp4"
        VideoUtilities.save_images_to_video(images=self.image_to_save_list, save_path=save_path, image_width=self.image_width, image_height=self.image_height)

    def save(self, folder_path: str):

        # Function used to save the 3D world coordinates and 2D image coordinates of each persons
        self.results = {}
        for i in range(self.n_frame):
            self.results[i] = {}
            for id in self.results_hollistic_w[i].keys():

                self.results[i][id] = {}
                self.results[i][id] = self.results_hollistic_w[i][id].tolist()

        JSONUtilities.save_dictionary_as_json(data_dict=self.results, folder_path=folder_path, filename="Hollistic_w")

        self.results = {}
        for i in range(self.n_frame):
            self.results[i] = {}
            for id in self.results_hollistic_i[i].keys():

                self.results[i][id] = {}
                self.results[i][id] = self.results_hollistic_i[i][id].tolist()

        JSONUtilities.save_dictionary_as_json(data_dict=self.results, folder_path=folder_path, filename="Hollistic_i")

        self.results = {}
        for i in range(self.n_frame):
            self.results[i] = {}
            for id in self.results_hollistic_v[i].keys():

                self.results[i][id] = {}
                self.results[i][id] = self.results_hollistic_v[i][id].tolist()

        JSONUtilities.save_dictionary_as_json(data_dict=self.results, folder_path=folder_path, filename="Hollistic_v")

        # Save pose landmark from mediapipe
        np.save(folder_path + "Hollistic_mp_pl", self.results_hollistic_mp_pl)

path_info = PathInfo()

video_names = ["Video_test_0.mp4", "Video_test_1.mp4", "Video_test_2.mp4"]
parameters = {"draw_pose": True, "draw_face": True, "draw_3D": True}

# Extract video first
data_frame_dicts = {}
n_frame = 10

for video_name in video_names:

    video = ProcessVideo(video_path=path_info.path_data_test + video_name)
    data_frame_dict = video.read(n_frame=n_frame)
    data_frame_dicts[video_name] = data_frame_dict

for video_name in video_names:

    data_frame_dict = data_frame_dicts[video_name]

    path_info.set_results_folder(name=Path(video_name).stem)

    for filename in ["IDs"]:

        # Load IDs and box coordinates for persons
        results = JSONUtilities.load_json_as_dictionary(folder_path=path_info.results_folder, filename=filename)
        results = JSONUtilities.list_dict_to_dict_list(dict_list=results)

        if filename == "IDs":
            parameters["face"] = False
            parameters["body"] = True
        elif filename == "Face_IDs":
            parameters["face"] = True
            parameters["body"] = False

        video_media_pipe = VideoMediaPipe(data_frame_dict=data_frame_dict, results=results, parameters=parameters)
        video_media_pipe.process()
        video_media_pipe.save_video(folder_path=path_info.results_folder, video_name=Path(video_name).stem)
        video_media_pipe.save(folder_path=path_info.results_folder)

