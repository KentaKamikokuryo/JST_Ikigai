from Classes.PathInfo import PathInfo
from Classes.Video import ProcessVideo
import cv2
from Kenta.TempClasses.DataManager import DataManagerDetection, DataManagerReidentification
from Kenta.TempClasses.VideoManager import VideoManager
from Kenta.TempClassesML.Factories import DetectorFactory
from Kenta.TempClassesML.Models import DetectionType
from Kenta.TempClasses.PathInfo import KentaPathInfo
from pathlib import Path

pathInfo = PathInfo()
video_folder = pathInfo.path_data_test
video_names = ["Experimental_Video_0", "Experimental_Video_1", "Experimental_Video_2", "Interview_Video_0"]
video_name = video_names[0]
video_format = ".mp4"

cap = cv2.VideoCapture(video_folder + video_name + video_format)
processVideo = ProcessVideo(video_path=video_folder + video_name + video_format)
data_frame_dict = processVideo.read(n_frame=200, resize_param=2)

DETECTION_CONFIDENCE_THRESHOLD = 0.3

# region video manager
PLOT_PARAMETERS = {"scale_image": 1,
                   "scale_text": 3,
                   "thickness": 2,
                   "margin": 20}

video_manager = VideoManager(plot_parameters=PLOT_PARAMETERS)
# endregion

# region process video using detection model
MODELS = [DetectionType.FACE_BOX_RF,
          DetectionType.FACE_BOX_OV,
          DetectionType.FACE_BOX_MP,
          DetectionType.BODY_BOX_OV,
          DetectionType.BODY_BOX_YOLO]

MODEL = MODELS[0]

fac = DetectorFactory(detection_type=MODEL,
                      path_model=pathInfo.path_model,
                      threshold=DETECTION_CONFIDENCE_THRESHOLD)

data_detection = DataManagerDetection(data_frame_dict=data_frame_dict,
                                      detector=fac.model)

data_detection.infer()
# endregion

# region draw bbox with confidence
_ = video_manager.write_bbox(data_frame_dict=data_frame_dict,
                             person_results=data_detection.person_results,
                             conf=data_detection.confidence_results,
                             fac=fac,
                             is_info=False)
video_manager.save_video(video_save_folder=KentaPathInfo.saved_video_path,
                         video_name=video_name,
                         model_names=MODEL)
# endregion

# region process bbox using Re-identification model
pathInfo.set_results_folder(name=Path(video_name).stem)

data_reid = DataManagerReidentification(data_frame_dict=data_frame_dict,
                                        detector=fac,
                                        path_model=pathInfo.path_model,
                                        path_info=pathInfo)

data_reid.infer(person_results=data_detection.person_results, confidence_results=data_detection.confidence_results)
data_reid.save(folder_path=pathInfo.results_folder)

# endregion

# region draw bbox and ids with confidence

_ = video_manager.write_bbox_ids(data_frame_dict=data_frame_dict,
                                 person_results=data_detection.person_results,
                                 ids_results=data_reid.ids_results,
                                 conf=data_detection.confidence_results)
video_manager.save_video(video_save_folder=KentaPathInfo.saved_video_path,
                         video_name=video_name,
                         model_names="{}_ReID".format(MODEL))

# endregion
