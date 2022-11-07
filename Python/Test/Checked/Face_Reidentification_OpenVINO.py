from ClassesML.FaceDetectionModel import FaceDetectionModel
from ClassesML.ReIdentification import ReIdentificationModel
from Classes.PathInfo import PathInfo
from Classes.Video import VideoUtilities
from Classes.Video import ProcessVideo
from Kenta.TempClasses.PathInfo import KentaPathInfo
import cv2
import copy
import numpy as np
import random

class FaceDetector(FaceDetectionModel):

    def __init__(self, path_model):

        self.model_name = "face-detection-retail-0005"
        self._conf = []

        super().__init__(path_model=path_model, model_name=self.model_name)

    def infer(self, image):

        scaled_frame, scale_h, scale_w = self.scale_frame(frame=image)

        infer_result = self.compiled_model([scaled_frame])[self.output_layer]

        detections = self._get_detection(infer_result=infer_result,
                                         image=image)

        person_faces = self._get_person_faces(detections=detections)

        return detections, person_faces

    def _get_person_faces(self, detections):

        person_faces = []

        if (len(detections) > 0):

            print("-------------------")
            for detection in detections:
                x1 = int(detection[0])
                y1 = int(detection[1])
                x2 = int(detection[2])
                y2 = int(detection[3])
                conf = detection[4]
                print("{:.1f} ({},{})-({},{})".format(conf, x1, y1, x2, y2))
                person_faces.append([x1, y1, x2, y2])
                self._conf.append(conf)

        return person_faces

    def _get_detection(self, infer_result, image):

        detections = []
        height, width = image.shape[:2]
        for r in infer_result[0][0]:
            conf = r[2]
            if (conf > self.threshold):
                x1 = int(r[3] * width)
                y1 = int(r[4] * height)
                x2 = int(r[5] * width)
                y2 = int(r[6] * height)
                detections.append([x1, y1, x2, y2, conf])

        return detections

    @property
    def conf(self):
        return self._conf

class FaceReidentification(ReIdentificationModel):

    def __init__(self, path_model):

        self.model_name = "face-reidentification-retail-0095"

        super().__init__(path_model=path_model,
                         model_name=self.model_name)

    def infer(self, image, person_faces):

        identifies = np.zeros((len(person_faces), 256))

        for i, face in enumerate(person_faces):

            img = image[face[1]: face[3], face[0]: face[2]]
            h, w = img.shape[:2]

            if (h == 0 or w == 0):
                continue

            scaled_img, scale_h, scale_w = self.scale_frame(img)

            results = self.compiled_model([scaled_img])[self.output_layer]
            # np.delete(results, 1)
            identifies[i] = results[0, :, 0, 0]

        return identifies

class Tracker:
    def __init__(self):
        # Database of identification information
        self.identifysDb = None
        # DB of face confidence
        self.conf = []

    def getIds(self, identifys, detections):
        if (identifys.size == 0):
            return []
        if self.identifysDb is None:
            self.identifysDb = identifys
            for person in detections:
                self.conf.append(person[4])

        print("input: {} DB:{}".format(len(identifys), len(self.identifysDb)))
        similaritys = self.__cos_similarity(identifys, self.identifysDb)
        similaritys[np.isnan(similaritys)] = 0
        ids = np.nanargmax(similaritys, axis=1)

        for i, similarity in enumerate(similaritys):
            persionId = ids[i]
            print("persionId:{} {} conf:{}".format(persionId, similarity[persionId], detections[i][4]))
            # If 0.9 or higher and the face detection confidence is higher than the existing one, the identification information is updated
            if (similarity[persionId] > 0.9 and detections[i][4] > self.conf[persionId]):
                print("? refresh id:{} conf:{}".format(persionId, detections[i][4]))
                self.identifysDb[persionId] = identifys[i]
            # If less than 0.3, add
            elif (similarity[persionId] < 0.3):
                self.identifysDb = np.vstack((self.identifysDb, identifys[i]))
                self.conf.append(detections[i][4])
                ids[i] = len(self.identifysDb) - 1
                print("append id:{} similarity:{}".format(ids[i], similarity[persionId]))

        print(ids)
        # If there are duplicates, disable the one with the lower confidence level (this is unlikely this time)
        for i, a in enumerate(ids):
            for e, b in enumerate(ids):
                if (e == i):
                    continue
                if (a == b):
                    if (similarity[a] > similarity[b]):
                        ids[i] = -1
                    else:
                        ids[e] = -1
        print(ids)
        return ids

    # Cosine similarity
    # reference document: https://github.com/kodamap/person_reidentification
    def __cos_similarity(self, X, Y):
        m = X.shape[0]
        Y = Y.T
        return np.dot(X, Y) / (
                np.linalg.norm(X.T, axis=0).reshape(m, 1) * np.linalg.norm(Y, axis=0)
        )


class Utils():

    @staticmethod
    def set_colors_tracking(tracking_max):

        colors = []

        for i in range(tracking_max):

            b = random.randint(0, 255)
            g = random.randint(0, 255)
            r = random.randint(0, 255)

            colors.append((b, g, r))

        return colors

class Manager():

    def __init__(self, data_frame_dict, path_model):

        self.data_frame_dict = data_frame_dict
        self.path_model = path_model

        self.n_frame = len(self.data_frame_dict.keys())

        self._get_models()
        self._set_results_dict()

    def process(self):

        self._detect_bbox()

        self._detect_tracks()

    def _get_models(self):

        self.face_detector = FaceDetector(path_model=self.path_model)

        self.face_re_identification = FaceReidentification(path_model=self.path_model)
        self.tracker = Tracker()

    def _set_results_dict(self):

        # Set dictionary for storing person face detection results
        self._person_face_results = {}
        self._detection_results = {}
        self._confidence_results = {}

        # for storing face re-identification results
        self._identifies_results = {}
        self._ids_results = {}

    def _detect_bbox(self):

        for i in range(self.n_frame):

            image = copy.deepcopy(self.data_frame_dict[i])

            self._detection_results[i], \
            self._person_face_results[i] = self.face_detector.infer(image=image)
            self._confidence_results[i] = self.face_detector.conf

            print("Processing face BBox at frame: " + str(i) + "/" + str(self.n_frame))

    def _detect_tracks(self):

        for i in range(self.n_frame):

            image = copy.deepcopy(self.data_frame_dict[i])

            self._identifies_results[i] = self.face_re_identification.infer(image=image,
                                                                            person_faces=self._person_face_results[i])

            self._ids_results[i] = self.tracker.getIds(identifys=self._identifies_results[i],
                                                       detections=self._detection_results[i])

            print("Processing ID at frame: " + str(i) + "/" + str(self.n_frame))

    @property
    def person_face_results(self):
        return self._person_face_results

    @property
    def detection_results(self):
        return self._detection_results

    @property
    def confidence_results(self):
        return self._confidence_results

    @property
    def identifies_results(self):
        return self._identifies_results

    @property
    def ids_results(self):
        return self._ids_results




plot_parameters = {"scale_image": 1,
                   "scale_text": 1.5,
                   "margin": 10}

class VideoManager():

    def __init__(self, data_frame_dict: dict, plot_parameters: dict):

        self.SCALE_IMAGE = plot_parameters["scale_image"]
        self.SCALE_TEXT = plot_parameters["scale_text"]
        self.MARGIN = plot_parameters["margin"]

        self.data_frame_dict = data_frame_dict
        self.n_frame = len(self.data_frame_dict.keys())

        self._image_list = []

    def write_bbox(self,
                   person_face_results: dict,
                   conf: dict,
                   video_save_folder: str,
                   video_name: str,
                   is_info: bool = False):

        self._image_list = []
        image_height = 0
        image_width = 0

        for i in range(self.n_frame):

            image = copy.deepcopy(self.data_frame_dict[i])
            image_height, image_width, _ = image.shape

            if is_info:

                SCALE_TEXT = 3

                image = cv2.putText(image, "Bounding Box: OpenVINO",
                                    (0, 50),
                                    cv2.FONT_HERSHEY_PLAIN,
                                    SCALE_TEXT,
                                    (255, 0, 0),
                                    SCALE_TEXT,
                                    cv2.LINE_AA)

            image = self._draw_bbox(image=image,
                                    faces=person_face_results[i],
                                    conf=conf[i],
                                    color=(255, 0, 0),
                                    is_write_conf=True)

            self._image_list.append(image)
            print("Drawing BBox at frame: " + str(i) + "/" + str(self.n_frame))

        save_path = video_save_folder + video_name + "bbox_face_OpenVINO.mp4"
        VideoUtilities.save_images_to_video(images=self._image_list,
                                            save_path=save_path,
                                            image_height=image_height,
                                            image_width=image_width)

    def write_bbox_tracks(self,
                          person_face_results: dict,
                          conf: dict,
                          ids: dict,
                          video_save_folder: str,
                          video_name: str,
                          is_info: bool = False):

        self._image_list = []
        image_height = 0
        image_width = 0
        colors = Utils.set_colors_tracking(tracking_max=50)

        for i in range(self.n_frame):

            image = copy.deepcopy(self.data_frame_dict[i])
            image_height, image_width, _ = image.shape

            if is_info:

                SCALE_TEXT = 3

                image = cv2.putText(image, "Bounding Box: OpenVINO",
                                    (0, 50),
                                    cv2.FONT_HERSHEY_PLAIN,
                                    SCALE_TEXT,
                                    (255, 0, 0),
                                    SCALE_TEXT,
                                    cv2.LINE_AA)

                image = cv2.putText(image, "Tracks (IDs): OpenVINO",
                                    (0, 100),
                                    cv2.FONT_HERSHEY_PLAIN,
                                    SCALE_TEXT,
                                    (255, 0, 0),
                                    SCALE_TEXT,
                                    cv2.LINE_AA)

            image = self._draw_tracks(image=image,
                                      faces=person_face_results[i],
                                      conf=conf[i],
                                      ids_results=ids[i],
                                      colors=colors)

            self._image_list.append(image)
            print("Drawing BBox and Tracks at frame: " + str(i) + "/" + str(self.n_frame))

        save_path = video_save_folder + video_name + "bbox_tracks_face_OpenVINO.mp4"
        VideoUtilities.save_images_to_video(images=self._image_list,
                                            save_path=save_path,
                                            image_height=image_height,
                                            image_width=image_width)

    def _draw_bbox(self, image, faces, conf, color, is_write_conf: bool = False):

        for f, face in enumerate(faces):

            start_point = (face[0] - self.MARGIN, face[1] - self.MARGIN)
            end_point = (face[2] + self.MARGIN, face[3] + self.MARGIN)

            image = cv2.rectangle(image,
                                  pt1=start_point,
                                  pt2=end_point,
                                  color=color,
                                  thickness=2)

            if is_write_conf:

                image = cv2.putText(image,
                                    text="confidence: {:.2f}".format(conf[f]),
                                    org=start_point,
                                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                                    fontScale=self.SCALE_TEXT,
                                    color=(0, 255, 0),
                                    lineType=cv2.LINE_AA)

        return image

    def _draw_tracks(self, image, faces, conf, ids_results, colors):

        for i, face in enumerate(faces):

            color = colors[int(ids_results[i])]

            start_point = (face[0] - self.MARGIN, face[1] - self.MARGIN)
            end_point = (face[2] + self.MARGIN, face[3] + self.MARGIN)

            image = cv2.rectangle(image,
                                  pt1=start_point,
                                  pt2=end_point,
                                  color=color,
                                  thickness=2)

            image = cv2.putText(image,
                                text="{}: {:.2f}".format(ids_results[i], conf[i]),
                                org=(face[0], face[1]),
                                fontFace=cv2.FONT_HERSHEY_PLAIN,
                                fontScale=self.SCALE_TEXT,
                                color=color,
                                lineType=cv2.LINE_AA)
        return image


pathInfo = PathInfo()
video_folder = pathInfo.path_data_test
video_name = "Video_test_0"
video_format = ".mp4"

cap = cv2.VideoCapture(video_folder + video_name + video_format)
processVideo = ProcessVideo(video_path=video_folder + video_name + video_format)
data_frame_dict = processVideo.read(n_frame=300)

manager = Manager(data_frame_dict=data_frame_dict,
                  path_model=pathInfo.path_model)
manager.process()

videoManager = VideoManager(data_frame_dict=data_frame_dict,
                            plot_parameters=plot_parameters)

videoManager.write_bbox(person_face_results=manager.person_face_results,
                        conf=manager.confidence_results,
                        video_save_folder=KentaPathInfo().saved_video_path,
                        video_name=video_name,
                        is_info=True)

videoManager.write_bbox_tracks(person_face_results=manager.person_face_results,
                               conf=manager.confidence_results,
                               ids=manager.ids_results,
                               video_save_folder=KentaPathInfo().saved_video_path,
                               video_name=video_name,
                               is_info=True)
