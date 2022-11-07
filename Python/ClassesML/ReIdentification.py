import copy

from Classes.ProcessWebcam import *
import cv2
from ClassesML.ModelMLUtilities import *
from ClassesDB.DatabaseIDs import *

class ReIdentificationModel:

    def __init__(self, path_model, model_name):

        self.device = "CPU"
        self.num_requests = 2
        self.threshold = 0.6

        self.model, self.compiled_model = ModelMLUtilities.load_local_xml_model(path_model + model_name, device_name=self.device)

        ModelMLUtilities.display_input_onnx_model_information(self.model)
        ModelMLUtilities.display_output_onnx_model_information(self.model)

        self.input_name = ModelMLUtilities.get_input_name_onnx_model(self.model)
        self.output_name = ModelMLUtilities.get_output_name_onnx_model(self.model)

        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)

        self.input_size = self.compiled_model.input(0).shape
        self.output_size = self.compiled_model.output(0).shape

        self.input_height = self.input_size[2]
        self.input_width = self.input_size[3]

    def scale_frame(self, frame):

        initial_h, initial_w = frame.shape[:2]
        scale_h, scale_w = initial_h / float(self.input_height), initial_w / float(self.input_width)
        in_frame = cv2.resize(frame, (self.input_width, self.input_height))
        in_frame = in_frame.transpose((2, 0, 1))
        in_frame = in_frame.reshape(self.input_size)

        return in_frame, scale_h, scale_w

class PersonBodyReidentification(ReIdentificationModel):

    def __init__(self, path_model):

        self.model_name = "person-reidentification-retail-0288"

        super().__init__(path_model, self.model_name)

    def infer(self, image, persons):

        self.image = image

        identifies = np.zeros((len(persons), 256))
        image_boxes = []

        for i, person in enumerate(persons):

            # Acquisition of each person's image
            img = image[person[1]: person[3], person[0]: person[2]]
            h, w = img.shape[:2]

            if (h == 0 or w == 0):
                continue

            scaled_img, self.scale_h, self.scale_w = self.scale_frame(img)

            # Identification information acquisition
            results = self.compiled_model([scaled_img])[self.output_layer]
            identifies[i] = results[0, :]

            image_boxes.append(copy.deepcopy(img))

        return identifies, image_boxes

class PersonFaceReidentification(ReIdentificationModel):

    def __init__(self, path_model):

        self.model_name = "face-reidentification-retail-0095"

        super().__init__(path_model=path_model, model_name=self.model_name)

    def infer(self, image, persons):

        identifies = np.zeros((len(persons), 256))
        image_boxes = []

        for i, face in enumerate(persons):

            img = image[face[1]: face[3], face[0]: face[2]]
            h, w = img.shape[:2]

            if (h == 0 or w == 0):
                continue

            scaled_img, scale_h, scale_w = self.scale_frame(img)

            results = self.compiled_model([scaled_img])[self.output_layer]
            identifies[i] = results[0, :, 0, 0]

            image_boxes.append(copy.deepcopy(img))

        return identifies, image_boxes
