import torch
from openvino.runtime import Core
import os
from torchvision import models
import sys

class ModelMLUtilities:

    @staticmethod
    def load_online_pytorch_model(url: str, path_model: str):

        # url: url of the model
        # path_model: path to where the downloaded model will be saved

        # Create device
        device = torch.device('cpu')

        # Load model
        model = torch.hub.load(url, 'custom', path_model)
        model.to(device)

        return model

    @staticmethod
    def load_local_pytorch_model(path_model: str):

        device = torch.device("cpu")
        # sys.path.insert(0, path_model + ".pt")
        # print(sys.path)

        print("The pytorch model is loaded from {}".format(path_model + ".pt"))

        # model = bentoml.pytorch.load(path_model + ".pt")

        model = models.mnasnet0_5()
        weights = torch.load(path_model + ".pt")
        model.load_state_dict(weights)
        model.to(device=device)

        return model

    @staticmethod
    def export_pytorch_model_as_onnx(model, path_model: str):

        # Model: pytorch model
        # path_model: full path of the model to save (without extension)

        device = torch.device('cpu')

        # Useless input (Necessary for the export phase)
        dummy_input = torch.randn(1, 3, 384, 640, device=device)

        dynamic_axes = {'images': [0, 2, 3]}
        input_names = ['images']

        # Export as onnx
        torch.onnx.export(model, (dummy_input,), input_names=input_names, dynamic_axes=dynamic_axes, f=path_model + '.onnx')

        # Example: https://github.com/pytorch/pytorch/blob/326d777e5384a621306330b5af0f2857843fe544/test/onnx/test_operators.py#L277

    @staticmethod
    def load_local_onnx_model(path_model: str):

        # path_model: full path of the model to load (without extension)

        if os.path.exists(path_model + ".onnx"):

            ie = Core()

            model = ie.read_model(model=path_model + ".onnx")

        else:
            model = None
            print("Cannot find model at: " + path_model + ".onnx")

        return model

    @staticmethod
    def load_local_xml_model(path_model: str, device_name: str = "CPU"):

        # path_model: full path of the model to load (without extension)

        if os.path.exists(path_model + ".xml"):

            ie = Core()

            model = ie.read_model(path_model + ".xml", path_model + ".bin")
            model_compiled = ie.compile_model(model, device_name=device_name)

        else:
            model = None
            model_compiled = None

            print("Cannot find model at: " + path_model + ".xml")

        return model, model_compiled

    @staticmethod
    def get_input_name_onnx_model(model):

        input_layer = model.input(0)
        input_layer_name = input_layer.any_name
        return input_layer_name

    @staticmethod
    def get_output_name_onnx_model(model):

        output_layer = model.output(0)
        output_layer_name = output_layer.any_name
        return output_layer_name

    @staticmethod
    def display_input_onnx_model_information(model):

        input_layer = model.input(0)
        input_layer_name = input_layer.any_name
        input_precision = input_layer.element_type
        input_shape = input_layer.shape

        print("Model input: " + input_layer_name + " with shape" + str(input_shape) + " and precision: " + str(input_precision))

    @staticmethod
    def display_output_onnx_model_information(model):

        output_layer = model.output(0)
        output_layer_name = output_layer.any_name
        output_precision = output_layer.element_type
        output_shape = output_layer.shape

        print("Model output: " + output_layer_name + " with shape" + str(output_shape) + " and precision: " + str(output_precision))

    @staticmethod
    def useless_for_now_doing_prediction_on_image_with_onnx_model():

        # To make this work will require to code our own auto shape model

        # Test on one image
        image = data_frame_dict[0]
        image_height, image_width, image_channel = image.shape

        model_name = "yolov5s"

        yolov5s = ModelMLUtilities.load_local_onnx_model(path_model=path_info.path_model + model_name)

        ie = Core()
        compiled_model = ie.compile_model(model=yolov5s, device_name="CPU")
        input_layer = compiled_model.input(0)

        # current input size
        ModelMLUtilities.display_input_onnx_model_information(model=compiled_model)

        model = self.yolov5s

        # Reshape input for current image size
        self.yolov5s.reshape([1, image_channel, image_width, image_height])

        # since we are only interested in detecting person
        self.yolov5s.classes = [0]

        # Each image dimension must be an even multiple of 32 for P5 models
        ie = Core()
        compiled_model = ie.compile_model(model=model, device_name="CPU")

        # current input size
        ModelMLUtilities.display_input_onnx_model_information(model=compiled_model)

        image_reshape = np.transpose(image, (2, 0, 1))[np.newaxis]

        result = compiled_model([image_reshape])
