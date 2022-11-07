from Classes.PathInfo import PathInfo
from Classes.Video import ProcessVideo
from Classes.JSONUtilities import JSONUtilities
import cv2
from Kenta.TempClasses.DataManager import DataManagerEmotions
from Kenta.TempClasses.VideoManager import VideoManager
from Kenta.TempClassesML.Factories import FaceEmotionsDetectorFactory
from Kenta.TempClassesML.Models import FaceEmotionsDetectionType
from Kenta.TempClasses.PathInfo import KentaPathInfo
from pathlib import Path

pathInfo = PathInfo()
video_folder = pathInfo.path_data_test
video_names = ["Experimental_Video_0", "Experimental_Video_1", "Experimental_Video_2", "Interview_Video_0"]
video_name = video_names[3]
video_format = ".mp4"

cap = cv2.VideoCapture(video_folder + video_name + video_format)
processVideo = ProcessVideo(video_path=video_folder + video_name + video_format)
data_frame_dict = processVideo.read(n_frame=200, resize_param=2)

pathInfo.set_results_folder(name=Path(video_name).stem)

MODELS = [FaceEmotionsDetectionType.EMOTIONS_OV]

MODEL = MODELS[0]

fac = FaceEmotionsDetectorFactory(detection_type=MODEL,
                                  path_model=pathInfo.path_model)

FILE_NAME = "Face_IDs"

results = JSONUtilities.load_json_as_dictionary(folder_path=pathInfo.results_folder, filename=FILE_NAME)
results = JSONUtilities.list_dict_to_dict_list(dict_list=results)

data_manager = DataManagerEmotions(data_frame_dict=data_frame_dict,
                                   results=results,
                                   emotion_detector=fac.model)
data_manager.infer()

JSONUtilities.save_dictionary_as_json(data_dict=data_manager.results,
                                      folder_path=pathInfo.results_folder,
                                      filename=fac.json_name)
print("The MediaPipe results are saved to: {}".format(pathInfo.results_folder + fac.json_name + ".json"))

