import matplotlib
matplotlib.use('Qt5Agg')
import cv2
from pathlib import Path
from Classes.Video import ProcessVideo
from Classes.PathInfo import PathInfo
from Classes.JSONUtilities import JSONUtilities
import numpy as np
from Kenta.TempClasses.DataManager import DataManagerBiomechanics


pathInfo = PathInfo()
video_folder = pathInfo.path_data_test
video_names = ["Experimental_Video_0", "Experimental_Video_1", "Experimental_Video_2", "Interview_Video_0"]
video_name = video_names[1]
video_format = ".mp4"

cap = cv2.VideoCapture(video_folder + video_name + video_format)
processVideo = ProcessVideo(video_path=video_folder + video_name + video_format)
data_frame_dict = processVideo.read(n_frame=200, resize_param=2)

DETECTION_CONFIDENCE_THRESHOLD = 0.3

pathInfo.set_results_folder(name=Path(video_name).stem)

FILE_NAME = "IDs"

results = JSONUtilities.load_json_as_dictionary(folder_path=pathInfo.results_folder, filename=FILE_NAME)
results = JSONUtilities.list_dict_to_dict_list(dict_list=results)

data_manager = DataManagerBiomechanics(data_frame_dict=data_frame_dict,
                                       results=results,
                                       pathInfo=pathInfo)
data_manager.infer()

JSONUtilities.save_dictionary_as_json(data_dict=data_manager.results_b,
                                      folder_path=pathInfo.results_folder,
                                      filename=FILE_NAME + "_b")

print("The MediaPipe results are saved to: {}".format(pathInfo.results_folder + FILE_NAME + "_b.json"))

