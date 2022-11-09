from Classes.Hyperparameters import DetectionType
from Classes.Video import ProcessVideo
from ClassesML.ReIdentification import PersonFaceReidentification, PersonBodyReidentification
from ClassesML.FaceDetectionModel import *
from ClassesML.BodyDetectionModel import *
from ClassesDB.DatabaseSupporters import Database

class ModelFactory:

    def __init__(self, detection_type: DetectionType, path_model: str, threshold: float = 0.8):

        self.detection_type = detection_type
        self.path_model = path_model
        self.threshold = threshold

        self._create_model()

    def _create_model(self):

        if self.detection_type == DetectionType.BODY_BOX_YOLO:

            self._model_name = "Body_YOLO"
            self._model_type = DetectionType.BODY_BOX_YOLO
            self._model = PersonBodyDetectionYOLO(path_model=self.path_model,
                                                  threshold=self.threshold)
            self._reid = PersonBodyReidentification(self.path_model)

        elif self.detection_type == DetectionType.BODY_BOX_OV:

            self._model_name = "Body_OV"
            self._model_type = DetectionType.BODY_BOX_OV
            self._model = PersonBodyDetectorOV(path_model=self.path_model,
                                               threshold=self.threshold)
            self._reid = PersonBodyReidentification(self.path_model)

        elif self.detection_type == DetectionType.FACE_BOX_RF:

            self._model_name = "Face_RF"
            self._model_type = DetectionType.FACE_BOX_RF
            self._model = PersonFaceDetectorRF(path_model=self.path_model,
                                               threshold=self.threshold)
            self._reid = PersonFaceReidentification(self.path_model)

        elif self.detection_type == DetectionType.FACE_BOX_OV:

            self._model_name = "Face_OV"
            self._model_type = DetectionType.FACE_BOX_OV
            self._model = PersonFaceDetectorOV(path_model=self.path_model,
                                               threshold=self.threshold)
            self._reid = PersonFaceReidentification(self.path_model)

        elif self.detection_type == DetectionType.FACE_BOX_MP:

            self._model_name = "Face_MP"
            self._model_type = DetectionType.FACE_BOX_MP
            self._model = PersonFaceDetectorMP(path_model=self.path_model,
                                               threshold=self.threshold)
            self._reid = PersonFaceReidentification(self.path_model)

    @property
    def model(self):
        return self._model

    @property
    def model_name(self):
        return self._model_name

    @property
    def re_id(self):
        return self._reid


class BBoxReId:

    def __init__(self, data_frame_dict: dict, fac: ModelFactory):

        self.data_frame_dict = data_frame_dict
        self.n_frame = len(self.data_frame_dict.keys())
        self.fac = fac

        self._set_model()
        self._set_results_dict_bbox()
        self._set_results_dict_reid()

    def infer(self):

        self._detect_faces()
        self._detect_ids()

    def _set_results_dict_bbox(self):

        self._person_results = {}
        self._confidence_results = {}

    def _set_results_dict_reid(self):

        self._identifies_results = {}
        self._image_boxes = {}

    def _set_model(self):

        self.model = self.fac.model
        self.re_id = self.fac.re_id

    def _detect_faces(self):

        for i in range(self.n_frame):

            image = copy.deepcopy(self.data_frame_dict[i])

            _, self._person_results[i] = self.model.infer(image=image)
            self._confidence_results[i] = self.model.conf

            # Obtain only those with a highest level of confidence
            if len(self._confidence_results[i]) > 1:

                idx = np.argsort(self._confidence_results[i])
                self._person_results[i] = [self._person_results[i][idx[0]]]
                self._confidence_results[i] = [self._confidence_results[i][idx[0]]]

    def _detect_ids(self):

        for i in range(self.n_frame):

            print("")
            print("Reidentification at frame: " + str(i))

            image = copy.deepcopy(self.data_frame_dict[i])
            person_result = copy.deepcopy(self._person_results[i])

            self._identifies_results[i], self._image_boxes[i] = self.re_id.infer(image=image,
                                                                                 persons=person_result)

    @property
    def person_results(self):
        return self._person_results

    @property
    def confidence_results(self):
        return self._confidence_results

    @property
    def identifies_results(self):
        return self._identifies_results

    @property
    def image_boxes(self):
        return self._image_boxes


path_info = PathInfo()
database_video_names = ["Seiko", "Tomoko", "Takashi"]
video_format = ".mp4"

detection_types_face = [DetectionType.FACE_BOX_OV, DetectionType.FACE_BOX_RF, DetectionType.FACE_BOX_MP]
detection_type_face = detection_types_face[0]

detection_types_body = [DetectionType.BODY_BOX_OV, DetectionType.BODY_BOX_YOLO]
detection_type_body = detection_types_body[0]

DETECTION_THRESHOLD = 0.5
REIDENTIFICATION_THRESHOLD = 0.9
N_FRAME = 50

#########################################
# Processing for extracting the face id #
#########################################
print("FACE")
for d, video_name in enumerate(database_video_names):

    data_frame_dict_supporters = {}

    print("Getting the video data: {}".format(video_name))
    # Set video for face
    print("")
    print("Extracting video frame of {}".format(video_name))
    video = ProcessVideo(video_path=path_info.path_database_video_face + video_name + video_format)
    ALL_N_FRAME = copy.deepcopy(video.n_frame)
    data_frame_dict_face = video.read(n_frame=ALL_N_FRAME)

    # Set Model for face and Infer
    fac_face = ModelFactory(detection_type=detection_type_face,
                            path_model=path_info.path_model,
                            threshold=DETECTION_THRESHOLD)
    box_id_face = BBoxReId(data_frame_dict=data_frame_dict_face, fac=fac_face)
    box_id_face.infer()

    # Set Class for getting database
    data_base_face = Database(path_database_supp=path_info.path_database_video_fid, name=video_name, id=d)

    for i in range(len(data_frame_dict_face.keys())):

        data_base_face.get_id_simple(identify=box_id_face.identifies_results[i], image_box=box_id_face.image_boxes[i])
    # TODO: Select one frame as parameter
    data_base_face.save(image_frame_per_person=0)

#########################################
# Processing for extracting the body id #
#########################################
print("BODY")
for d, video_name in enumerate(database_video_names):

    data_frame_dict_supporters = {}

    print("Getting the video data: {}".format(video_name))
    # Set video for body
    print("")
    print("Extracting video frame of {}".format(video_name))
    video = ProcessVideo(video_path=path_info.path_database_video_body + video_name + video_format)
    ALL_N_FRAME = copy.deepcopy(video.n_frame)
    data_frame_dict_body = video.read(n_frame=ALL_N_FRAME)

    # Set Model for body and Infer
    fac_body = ModelFactory(detection_type=detection_type_body,
                            path_model=path_info.path_model,
                            threshold=DETECTION_THRESHOLD)
    box_id_body = BBoxReId(data_frame_dict=data_frame_dict_body, fac=fac_body)
    box_id_body.infer()

    data_base_body = Database(path_database_supp=path_info.path_database_video_bid, name=video_name, id=d)

    for i in range(len(data_frame_dict_body.keys())):

        data_base_body.get_id_simple(identify=box_id_body.identifies_results[i], image_box=box_id_body.image_boxes[i])

    data_base_body.save(image_frame_per_person=0)


