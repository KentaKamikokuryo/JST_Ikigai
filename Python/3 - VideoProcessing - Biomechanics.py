import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from Classes.PathInfo import PathInfo
from Classes.Video import *
import cv2
import json
import random
from ClassesML.ReIdentification import *
from Classes.Hyperparameters import *
from Classes.JSONUtilities import *
from Classes.Pose import *

path_info = PathInfo()

class MediaPipeBiomechanics:

    def __init__(self, data_frame_dict, parameters):

        self.data_frame_dict = data_frame_dict

        # Load IDs and box coordinates for persons
        results = JSONUtilities.load_json_as_dictionary(folder_path=path_info.results_folder, filename="IDs")
        results = JSONUtilities.list_dict_to_dict_list(dict_list=results)

        # Load 3D and 2D hollistics coordinates
        self.results_hollistic_i = JSONUtilities.load_json_as_dictionary(folder_path=path_info.results_folder,
                                                                         filename="Hollistic_i")
        self.results_hollistic_w = JSONUtilities.load_json_as_dictionary(folder_path=path_info.results_folder,
                                                                         filename="Hollistic_w")
        self.results_hollistic_v = JSONUtilities.load_json_as_dictionary(folder_path=path_info.results_folder,
                                                                         filename="Hollistic_v")
        self.results_hollistic_mp_pl = np.load(path_info.results_folder + "Hollistic_mp_pl.npy", allow_pickle=True).item()

        self.persons_results = results["Box"]
        self.ids_results = results["IDs"]

        self.draw_pose = parameters["draw_pose"]
        self.draw_face = parameters["draw_face"]
        self.draw_3D = parameters["draw_3D"]

        self.n_frame = len(self.results_hollistic_w)

        self.results_b = {}  # b - Biomechanics
        self.results_b_i = {}  # b - Biomechanics on image

        self.get_models()

        self.MARGIN = 20

    def get_models(self):

        self.process_holistic_frame = ProcessHolisticFrame()
        self.process_holistic_frame.set_parameters()

    def process(self):

        for i in range(self.n_frame):

            self.results_b[i] = {}
            self.results_b_i[i] = {}

            # Goes through each person
            persons_results = self.persons_results[i]
            ids_results = self.ids_results[i]

            for j, person in enumerate(persons_results):

                id = ids_results[j]

                if (id != -1):

                    results_v = np.array(self.results_hollistic_v[i][str(id)]).squeeze().tolist()
                    results_hollistic_w = np.array(self.results_hollistic_w[i][str(id)]).squeeze()
                    results_hollistic_i = np.array(self.results_hollistic_i[i][str(id)]).squeeze()

                    presenceVisibility = PresenceVisibility(results_v=results_v, threshold=0.6)
                    biomechanicsInformation = BiomechanicsInformation(results_hollistic_w=results_hollistic_w,
                                                                      presenceVisibility=presenceVisibility,
                                                                      display=False)
                    biomechanicsImageInformation = BiomechanicsImageInformation(results_i=results_hollistic_i)

                    self.results_b[i][id] = biomechanicsInformation.dict
                    self.results_b_i[i][id] = biomechanicsImageInformation.dict

    def save_video(self, folder_path, video_name):

        self.image_to_save_list = []

        i = 0

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

                    x_pos = person[0]
                    y_pos = person[1]

                    if (h == 0 or w == 0):
                        continue

                    if self.draw_pose:
                        if self.results_hollistic_mp_pl[i][id] is not None:
                            DrawingUtilities.draw_pose_pl(image=img, results_mp_pl=self.results_hollistic_mp_pl[i][id], mp_holistic=self.process_holistic_frame.mp_holistic)

                    # Draw angles
                    for key in self.results_b_i[i][id].keys():

                        if self.results_b[i][id][key]:
                            value = round(self.results_b[i][id][key], 1)

                        if "r" in key[-1]:
                            x = self.results_b_i[i][id][key][0] - 80
                            y = self.results_b_i[i][id][key][1]

                        else:
                            x = self.results_b_i[i][id][key][0] + 20
                            y = self.results_b_i[i][id][key][1]

                        cv2.putText(img=image,
                                    text=str(value),
                                    org=(int(x) + x_pos, int(y) + y_pos),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.5,
                                    color=Drawing.color_red,
                                    thickness=1,
                                    lineType=cv2.LINE_AA)

                    cv2.imshow("", image)

            self.image_to_save_list.append(image)
            print("Drawing hollistic pose at frame: " + str(i) + "/" + str(self.n_frame))

        # Save video
        save_path = folder_path + video_name + "_" + "biomechanics" + ".mp4"
        VideoUtilities.save_images_to_video(images=self.image_to_save_list, save_path=save_path, image_width=self.image_width, image_height=self.image_height)

    def save(self, folder_path: str):

        # Function used to save the 3D world coordinates and 2D image coordinates of each persons
        self.results = {}
        for i in range(self.n_frame):

            self.results[i] = {}
            ids_results = self.ids_results[i]

            for id in ids_results:

                self.results[i][id] = {}
                self.results[i][id] = self.results_b[i][id]

        JSONUtilities.save_dictionary_as_json(data_dict=self.results, folder_path=folder_path, filename="Hollistic_b")

path_info = PathInfo()

video_names = ["Video_test_0.mp4", "Video_test_1.mp4", "Video_test_2.mp4"]
parameters = {"draw_pose": True, "draw_face": True, "draw_3D": True, "box_detection": "YOLO+OV"}

video_name = video_names[0]

for video_name in video_names:

    path_info.set_results_folder(folder_path=path_info.path_data_test, name=Path(video_name).stem)

    # Get video as dictionary of image
    video = ProcessVideo(video_path=path_info.path_data_test + video_name)
    data_frame_dict = video.read(n_frame=100)

    biomechanics = MediaPipeBiomechanics(data_frame_dict=data_frame_dict, parameters=parameters)
    biomechanics.process()
    biomechanics.save_video(folder_path=path_info.results_folder, video_name=video.video_name)
    biomechanics.save(folder_path=path_info.results_folder)
    self = biomechanics