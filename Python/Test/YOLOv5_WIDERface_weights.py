import os
from ClassesML.ModelMLUtilities import *
from Classes.PathInfo import *

path_info = PathInfo()

model_name = "face_detection_yolov5s"

model = ModelMLUtilities.load_local_pytorch_model(path_model=path_info.path_model + model_name)
