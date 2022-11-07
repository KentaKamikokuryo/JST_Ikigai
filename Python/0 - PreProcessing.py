import os
import torch
from ClassesML.ModelMLUtilities import *
from Classes.PathInfo import *
from openvino.runtime import Core

##############################################################################
# Download YOLO pytorch model and save it as .onnx file to use with OpenVino #
##############################################################################
path_info = PathInfo()

model_name = "yolov5s"

# Download pytorch model
model = ModelMLUtilities.load_online_pytorch_model(url='ultralytics/yolov5', path_model=path_info.path_model + model_name)

# Load local model
# model = ModelMLUtilities.load_local_pytorch_model(path_model=path_info.path_model + model_name)

# Export pytorch model to .onnx
model = ModelMLUtilities.export_pytorch_model_as_onnx(model=model, path_model=path_info.path_model + model_name)

# Load .onnx model with openvino
model = ModelMLUtilities.load_local_onnx_model(path_model=path_info.path_model + model_name)

