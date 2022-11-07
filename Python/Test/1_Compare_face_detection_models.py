from Classes.PathInfo import PathInfo
from Classes.Video import ProcessVideo
import cv2
from Kenta.TempClasses.DataManager import DataManagerDetection
from Kenta.TempClasses.VideoManager import VideoManager
from Kenta.TempClassesML.Factories import DetectorFactory
from Kenta.TempClassesML.Models import DetectionType
from Kenta.TempClasses.PathInfo import KentaPathInfo

pathInfo = PathInfo()
video_folder = pathInfo.path_data_test
video_names = ["Experimental_Video_0", "Experimental_Video_1", "Experimental_Video_2", "Interview_Video_0"]
video_name = video_names[1]
video_format = ".mp4"

cap = cv2.VideoCapture(video_folder + video_name + video_format)
processVideo = ProcessVideo(video_path=video_folder + video_name + video_format)
data_frame_dict = processVideo.read(n_frame=2, resize_param=2)

DETECTION_CONFIDENCE_THRESHOLD = 0.3

# region face detection model
print("---------------------")
print("Build Face detection factory as RetineFace")
fac_rf = DetectorFactory(detection_type=DetectionType.FACE_BOX_RF,
                         path_model=pathInfo.path_model,
                         threshold=DETECTION_CONFIDENCE_THRESHOLD)

print("---------------------")
print("Build Face detection factory as OpenVINO")
fac_ov = DetectorFactory(detection_type=DetectionType.FACE_BOX_OV,
                         path_model=pathInfo.path_model,
                         threshold=DETECTION_CONFIDENCE_THRESHOLD)

print("---------------------")
print("Build Face detection factory as MediaPipe")
fac_mp = DetectorFactory(detection_type=DetectionType.FACE_BOX_MP,
                         path_model=pathInfo.path_model,
                         threshold=DETECTION_CONFIDENCE_THRESHOLD)
# endregion

# region detect bbox
print("---------------------")
print("Inferring - RetinaFace")
data_rf = DataManagerDetection(data_frame_dict=data_frame_dict,
                               detector=fac_rf.model)
data_rf.infer()

print("---------------------")
print("Inferring - OpenVINO")
data_ov = DataManagerDetection(data_frame_dict=data_frame_dict,
                               detector=fac_ov.model)
data_ov.infer()

print("---------------------")
print("Inferring - MediaPipe")
data_mp = DataManagerDetection(data_frame_dict=data_frame_dict,
                               detector=fac_mp.model)
data_mp.infer()
# endregion

# manage video
plot_parameters = {"scale_image": 1,
                   "scale_text": 3,
                   "thickness": 2,
                   "margin": 20}

video_manager = VideoManager(plot_parameters=plot_parameters)
print("---------------------")
processed_data_frame_dict = video_manager.write_bbox(data_frame_dict=data_frame_dict,
                                                     person_results=data_mp.person_results,
                                                     conf=data_mp.confidence_results,
                                                     fac=fac_mp,
                                                     is_info=True)

print("---------------------")
processed_data_frame_dict2 = video_manager.write_bbox(data_frame_dict=processed_data_frame_dict,
                                                      person_results=data_ov.person_results,
                                                      conf=data_ov.confidence_results,
                                                      fac=fac_ov,
                                                      is_info=True)

print("---------------------")
processed_data_frame_dict3 = video_manager.write_bbox(data_frame_dict=processed_data_frame_dict2,
                                                      person_results=data_rf.person_results,
                                                      conf=data_rf.confidence_results,
                                                      fac=fac_rf,
                                                      is_info=True)

video_manager.save_video(video_save_folder=KentaPathInfo.saved_video_path,
                         video_name=video_name,
                         model_names="rf_ov_mp")



