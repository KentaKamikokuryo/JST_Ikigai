import matplotlib
matplotlib.use('Qt5Agg')
import cv2
from pathlib import Path
from Classes.Video import ProcessVideo
from Classes.PathInfo import PathInfo
from Classes.JSONUtilities import JSONUtilities
from Kenta.TempClasses.DataManager import DataManagerFaceMediaPipe, DataManagerHolisticMediaPipe
from Kenta.TempClasses.VideoManager import VideoManager, VideoUtilities
from Kenta.TempClasses.PathInfo import KentaPathInfo
import copy
import numpy as np
import sys

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

FILE_NAMEs = ["IDs", "Face_IDs"]
FILE_NAME = FILE_NAMEs[0]

results = JSONUtilities.load_json_as_dictionary(folder_path=pathInfo.results_folder, filename=FILE_NAME)
results = JSONUtilities.list_dict_to_dict_list(dict_list=results)

# region PROCESS MEDIAPIPE

if FILE_NAME == "IDs":

    data_manager = DataManagerHolisticMediaPipe(data_frame_dict=data_frame_dict,
                                                results=results)

elif FILE_NAME == "Face_IDs":

    data_manager = DataManagerFaceMediaPipe(data_frame_dict=data_frame_dict,
                                            results=results)

else:

    sys.exit(1)

data_manager.process()

# endregion

# region VIDEO
PLOT_PARAMETERS = {"scale_image": 1,
                   "scale_text": 3,
                   "thickness": 2,
                   "margin": 20}

video_manager = VideoManager(plot_parameters=PLOT_PARAMETERS)

ids_conf_data_frame_dict = video_manager.write_bbox_ids(data_frame_dict=data_frame_dict,
                                                        person_results=results["Box"],
                                                        ids_results=results["IDs"],
                                                        conf=results["Conf"])

_ = video_manager.write_mediapipe_landmarks(data_frame_dict=ids_conf_data_frame_dict,
                                            person_results=results["Box"],
                                            ids_results=results["IDs"],
                                            mpl_results=data_manager.results_mpl,
                                            mp=data_manager.mp,
                                            file_name=FILE_NAME)

video_manager.save_video(video_save_folder=KentaPathInfo.saved_video_path,
                         video_name=video_name,
                         model_names="{}_MP".format(FILE_NAME))
# endregion

# region SAVE RESULTS
np.save(pathInfo.results_folder + FILE_NAME + "_mpl", data_manager.results_mpl)
print("The MediaPipe results are saved to: {}".format(pathInfo.results_folder + FILE_NAME + "_mpl.np"))

JSONUtilities.save_dictionary_as_json(data_dict=data_manager.results_w,
                                      folder_path=pathInfo.results_folder,
                                      filename=FILE_NAME + "_w")
print("The MediaPipe results are saved to: {}".format(pathInfo.results_folder + FILE_NAME + "_w.json"))

JSONUtilities.save_dictionary_as_json(data_dict=data_manager.results_i,
                                      folder_path=pathInfo.results_folder,
                                      filename=FILE_NAME + "_i")
print("The MediaPipe results are saved to: {}".format(pathInfo.results_folder + FILE_NAME + "_i.json"))

JSONUtilities.save_dictionary_as_json(data_dict=data_manager.results_v,
                                      folder_path=pathInfo.results_folder,
                                      filename=FILE_NAME + "_v")
print("The MediaPipe results are saved to: {}".format(pathInfo.results_folder + FILE_NAME + "_v.json"))
# endregion



