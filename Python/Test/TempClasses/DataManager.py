import copy
import numpy as np
from ClassesML.Tracker import Tracker
from ClassesML.ReIdentification import PersonBodyReidentification, PersonFaceReidentification
from Kenta.TempClassesML.Factories import DetectorFactory, FaceEmotionsDetectorFactory
from Kenta.TempClassesML.Models import DetectionType
from ClassesDB.DatabaseIDs import DatabaseIDs
from Kenta.TempClasses.PathInfo import KentaPathInfo
from Classes.JSONUtilities import JSONUtilities
from Kenta.TempClassesML.MediaPipeProcess import FaceMeshFrame, HolisticFrame
from Classes.PathInfo import PathInfo
from Classes.Pose import PresenceVisibility
from Classes.Pose import BiomechanicsInformation

class DataManager():

    def __init__(self, data_frame_dict, detector):

        self.data_frame_dict = data_frame_dict
        self.detector = detector

        self.n_frame = len(self.data_frame_dict.keys())

class DataManagerDetection(DataManager):

    def __init__(self, data_frame_dict, detector):

        self._set_results_dict()

        super().__init__(data_frame_dict=data_frame_dict, detector=detector)

    def infer(self):

        self._set_results_dict()

        self._detect_faces()

    def _set_results_dict(self):

        # Set dictionary for storing person face detection results
        self._person_results = {}
        self._detection_results = {}
        self._confidence_results = {}

    def _detect_faces(self):

        for i in range(self.n_frame):

            image = copy.deepcopy(self.data_frame_dict[i])

            self._detection_results[i], \
            self._person_results[i] = self.detector.infer(image=image)
            self._confidence_results[i] = self.detector.conf

            print("Processing face BBox at frame: " + str(i) + "/" + str(self.n_frame))

    @property
    def person_results(self):
        return self._person_results

    @property
    def detection_results(self):
        return self._detection_results

    @property
    def confidence_results(self):
        return self._confidence_results

class DataManagerReidentification(DataManager):

    def __init__(self, data_frame_dict, detector: DetectorFactory, path_model: str, path_info):

        self.path_model = path_model
        super().__init__(data_frame_dict=data_frame_dict, detector=detector)

        self.path_info = path_info
        self._set_results_dict()
        self._set_models()

    def _set_results_dict(self):

        # for storing face re-identification results
        self._identifies_results = {}  # Neural network output for person identification
        self._ids_results = {}  # IDs for this video
        self._image_boxes = {}  # Coordinates of box representing each person
        self._ids_db_results = {}  # IDs for database of person
        self._results = {}  # Results saved and used for MediaPipe on each person

    def _set_models(self):

        if self.detector.model_name == DetectionType.BODY_BOX_YOLO or \
           self.detector.model_name == DetectionType.BODY_BOX_OV:

            self.re_identification = PersonBodyReidentification(path_model=self.path_model)
            self.tracker = Tracker()
            self.path_database_id = self.path_info.path_database_id

        elif self.detector.model_name == DetectionType.FACE_BOX_MP or \
             self.detector.model_name == DetectionType.FACE_BOX_OV or \
             self.detector.model_name == DetectionType.FACE_BOX_RF:

            self.re_identification = PersonFaceReidentification(path_model=self.path_model)
            self.tracker = Tracker()
            self.path_database_id = self.path_info.path_database_face_id

        self.database_ids = DatabaseIDs(path_database_id=self.path_database_id)

    def infer(self, person_results: dict, confidence_results: dict):

        class Parameters():

            DETECTION_THRESHOLD_UP = 0.95
            DETECTION_THRESHOLD_DOWN = 0.5
            DISTANCE_THRESHOLD = 200
            DETECTION_N = 1

        for i in range(self.n_frame):

            print("")
            print("Re-identification at frame: {}/{}".format(i, self.n_frame))

            image = copy.deepcopy(self.data_frame_dict[i])
            person_result = copy.deepcopy(person_results[i])

            self._identifies_results[i], self._image_boxes[i] = self.re_identification.infer(image=image,
                                                                                             persons=person_result)
            self._ids_results[i] = self.tracker.getIds(identifies=self._identifies_results[i],
                                                       persons=person_result)
            self._ids_db_results[i] = self.database_ids.get_ids(identifies_new=self._identifies_results[i],
                                                                persons=person_result,
                                                                image_boxes=self._image_boxes[i],
                                                                DETECTION_THRESHOLD_UP=Parameters.DETECTION_THRESHOLD_UP,
                                                                DETECTION_THRESHOLD_DOWN=Parameters.DETECTION_THRESHOLD_DOWN,
                                                                DISTANCE_THRESHOLD=Parameters.DISTANCE_THRESHOLD,
                                                                DETECTION_N=Parameters.DETECTION_N)

            self._results[i] = {}
            self._results[i]["IDs"] = self._ids_db_results[i]
            self._results[i]["Box"] = person_result
            self._results[i]["Conf"] = copy.deepcopy(confidence_results[i])

    def save(self, folder_path: str = ""):

        print("")
        print("saved to : {}\{}".format(folder_path, self.detector.json_name+".json"))

        JSONUtilities.save_dictionary_as_json(data_dict=self._results,
                                              folder_path=folder_path,
                                              filename=self.detector.json_name)

    # region property
    @property
    def identifies_results(self):
        return self._identifies_results

    @property
    def ids_results(self):
        return self._ids_results

    @property
    def image_boxes(self):
        return self._image_boxes

    @property
    def ids_db_results(self):
        return self._ids_db_results

    @property
    def results(self):
        return self._results
    # endregion

class DataManagerFaceMediaPipe:

    def __init__(self, data_frame_dict: dict, results: dict):

        self.data_frame_dict = data_frame_dict
        self.persons_results = results["Box"]
        self.ids_results = results["IDs"]

        self.n_frame = len(self.data_frame_dict.keys())

        self._set_models()
        self._set_results_dict()

    def _set_models(self):

        self.face_mesh = FaceMeshFrame()
        self._mp = self.face_mesh.mp_face_mesh

    def _set_results_dict(self):

        self._results_face_mp = {}  # mp - MediaPipe
        self._results_face_w = {}  # w - 3D world coordinates
        self._results_face_i = {}  # i - 2D image coordinates
        self._results_face_v = {}  # v - Visibility
        self._results_face_mpl = {}

    def process(self):

        for i in range(self.n_frame):

            image = copy.deepcopy(self.data_frame_dict[i])

            self._results_face_mp[i] = self.face_mesh.infer(image=image,
                                                            person_face_results=self.persons_results[i],
                                                            ids_results=self.ids_results[i])
            self._results_face_w[i] = self.face_mesh.results_w
            self._results_face_i[i] = self.face_mesh.results_i
            self._results_face_v[i] = self.face_mesh.results_v
            self._results_face_mpl[i] = self.face_mesh.results_mpl

            print("Processing face mesh at frame: " + str(i) + "/" + str(self.n_frame))

    @property
    def results_mp(self):
        return self._results_face_mp

    @property
    def results_w(self):
        return self._results_face_w

    @property
    def results_i(self):
        return self._results_face_i

    @property
    def results_v(self):
        return self._results_face_v

    @property
    def results_mpl(self):
        return self._results_face_mpl

    @property
    def mp(self):
        return self._mp


class DataManagerHolisticMediaPipe:

    def __init__(self, data_frame_dict: dict, results: dict):

        self.data_frame_dict = data_frame_dict
        self.persons_results = results["Box"]
        self.ids_results = results["IDs"]

        self.n_frame = len(self.data_frame_dict.keys())

        self._set_models()
        self._set_results_dict()

    def _set_models(self):

        self.holistic = HolisticFrame()
        self._mp = self.holistic.mp_holistic

    def _set_results_dict(self):
        self._results_holistic_mp = {}
        self._results_holistic_w = {}
        self._results_holistic_i = {}
        self._results_holistic_v = {}
        self._results_holistic_mpl = {}  # mp_pl - MediaPipe - pose_landmarks

    def process(self):
        for i in range(self.n_frame):
            image = copy.deepcopy(self.data_frame_dict[i])

            self._results_holistic_mp[i] = self.holistic.infer(image=image,
                                                               person_results=self.persons_results[i],
                                                               ids_results=self.ids_results[i])
            self._results_holistic_w[i] = self.holistic.results_w
            self._results_holistic_i[i] = self.holistic.results_i
            self._results_holistic_v[i] = self.holistic.results_v
            self._results_holistic_mpl[i] = self.holistic.results_mpl

            print("Processing holistic at frame: {}/{}".format(i, self.n_frame))

    @property
    def results_mp(self):
        return self._results_holistic_mp

    @property
    def results_w(self):
        return self._results_holistic_w

    @property
    def results_i(self):
        return self._results_holistic_i

    @property
    def results_v(self):
        return self._results_holistic_v

    @property
    def results_mpl(self):
        return self._results_holistic_mpl

    @property
    def mp(self):
        return self._mp


class DataManagerBiomechanics():

    def __init__(self,
                 data_frame_dict: dict,
                 results: dict,
                 pathInfo: PathInfo):

        self.data_frame_dict = data_frame_dict
        self.n_frame = len(self.data_frame_dict.keys())

        self.persons_results = results["Box"]
        self.ids_results = results["IDs"]

        FILE_NAME = "IDs"

        self.results_w = JSONUtilities.load_json_as_dictionary(folder_path=pathInfo.results_folder,
                                                               filename=FILE_NAME + "_w")

        self.results_i = JSONUtilities.load_json_as_dictionary(folder_path=pathInfo.results_folder,
                                                               filename=FILE_NAME + "_i")

        self.results_v = JSONUtilities.load_json_as_dictionary(folder_path=pathInfo.results_folder,
                                                               filename=FILE_NAME + "_v")

    def _set_results_dict(self):

        self._results_b = {}
        self._results_b_i = {}

    def _compute_biomechanics(self):

        for i in range(self.n_frame):

            self._results_b[i] = {}
            self._results_b_i[i] = {}

            person_results = self.persons_results[i]
            ids_results = self.ids_results[i]

            for j, person in enumerate(person_results):

                id = ids_results[j]

                if (id != -1):

                    if self.results_v[i][str(id)]:

                        self.ids_results_v = np.array(self.results_v[i][str(id)])
                        self.ids_results_w = np.array(self.results_w[i][str(id)]).squeeze()

                    else:

                        self.ids_results_v = np.zeros((33, 1), float)
                        self.ids_results_w = np.zeros((33, 3), float)

                    presence_visibility = PresenceVisibility(results_v=self.ids_results_v, threshold=0.6)
                    biomechanics_information = BiomechanicsInformation(results_hollistic_w=self.ids_results_w,
                                                                       presenceVisibility=presence_visibility,
                                                                       display=True)

                    self._results_b[i][id] = biomechanics_information.dict

    def infer(self):

        self._set_results_dict()

        self._compute_biomechanics()

    @property
    def results_b(self):
        return self._results_b

    @property
    def results_b_i(self):
        return self._results_b_i


class DataManagerEmotions():

    def __init__(self, data_frame_dict: dict, results: dict, emotion_detector):

        self.data_frame_dict = data_frame_dict
        self.n_frame = len(data_frame_dict.keys())
        self.emotion_detector = emotion_detector

        self.persons_results = results["Box"]
        self.ids_results = results["IDs"]
        self.label = ('neutral', 'happy', 'sad', 'surprise', 'anger')

        self._set_results_dict()

    def _set_results_dict(self):

        self._emotion_results = {}
        self._emotion_name_results = {}
        self._emotion_prob_results = {}

    def infer(self):

        self._results = {}

        for i in range(self.n_frame):

            image = copy.deepcopy(self.data_frame_dict[i])

            self._emotion_results[i] = self.emotion_detector.infer(image=image,
                                                                   faces=self.persons_results[i])

            emotion_names = []
            emotion_prob = []

            for j in range(len(self._emotion_results[i])):

                temp = self.label[np.argmax(self._emotion_results[i][j].squeeze())]
                emotion_names.append(temp)
                prob = np.max(self._emotion_results[i][j].squeeze())
                emotion_prob.append(prob)

            self._emotion_name_results[i] = emotion_names
            self._emotion_prob_results[i] = emotion_prob

            self._results[i] = {}
            self._results[i]["IDs"] = self.ids_results[i]
            self._results[i]["Box"] = self.persons_results[i]
            self._results[i]["Emotions"] = self._emotion_name_results[i]
            self._results[i]["Emotion_prob"] = self._emotion_prob_results[i]

            print("Processing face emotion recognition: " + str(i) + "/" + str(self.n_frame))

    @property
    def emotion_results(self):
        return self._emotion_results

    @property
    def emotion_name_results(self):
        return self._emotion_name_results

    @property
    def emotion_prob_results(self):
        return self._emotion_prob_results

    @property
    def results(self):
        return self._results


