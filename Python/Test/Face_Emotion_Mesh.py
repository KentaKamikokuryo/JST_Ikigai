from ClassesML.ModelMLUtilities import *
from Classes.PathInfo import PathInfo
from Classes.Video import VideoUtilities
from Classes.Video import ProcessVideo
from Kenta.TempClasses.PathInfo import KentaPathInfo
from Classes.Drawing import DrawingUtilities, Drawing
import cv2
import copy
import numpy as np
import logging as log
import random
import mediapipe as mp

class Model:

    def __init__(self, path_model, model_name):

        self.device = "CPU"
        self.num_requests = 2
        self.threshold = 0.5

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

def put_highlighted_text(frame, message, position, font_face, font_scale, color, thickness):
    cv2.putText(frame, message, position, font_face, font_scale, (255, 255, 255), thickness + 1) # white border
    cv2.putText(frame, message, position, font_face, font_scale, color, thickness)

def draw_labels(frame, classifications, output_transform):

    frame = output_transform.resize(frame)
    class_label = ""
    if classifications:
        class_label = classifications[0][1]
    font_scale = 0.7
    label_height = cv2.getTextSize(class_label, cv2.FONT_HERSHEY_COMPLEX, font_scale, 2)[0][1]
    initial_labels_pos = frame.shape[0] - label_height * (int(1.5 * len(classifications)) + 1)

    if (initial_labels_pos < 0):
        initial_labels_pos = label_height
        log.warning('Too much labels to display on this frame, some will be omitted')
    offset_y = initial_labels_pos

    header = "Label:     Score:"
    label_width = cv2.getTextSize(header, cv2.FONT_HERSHEY_COMPLEX, font_scale, 2)[0][0]
    put_highlighted_text(frame, header, (frame.shape[1] - label_width, offset_y),
        cv2.FONT_HERSHEY_COMPLEX, font_scale, (255, 0, 0), 2)

    for idx, class_label, score in classifications:
        label = '{}. {}    {:.2f}'.format(idx, class_label, score)
        label_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, font_scale, 2)[0][0]
        offset_y += int(label_height * 1.5)
        put_highlighted_text(frame, label, (frame.shape[1] - label_width, offset_y),
            cv2.FONT_HERSHEY_COMPLEX, font_scale, (255, 0, 0), 2)
    return frame

class FaceDetector(Model):

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

class FaceReidentification(Model):

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

        self.center = []

    def _getCenter(self, person):
        x = person[0] - person[2]
        y = person[1] - person[3]
        return (x, y)

    def _getDistance(self, person, index):

        (x1, y1) = self.center[index]
        (x2, y2) = self._getCenter(person)
        a = np.array([x1, y1])
        b = np.array([x2, y2])
        u = b - a
        return np.linalg.norm(u)

    def getIds(self, identifys, detections):
        if (identifys.size == 0):
            return []
        if self.identifysDb is None:
            self.identifysDb = identifys
            for person in detections:
                self.center.append(self._getCenter(person))
                self.conf.append(person[4])

        print("input: {} DB:{}".format(len(identifys), len(self.identifysDb)))
        similaritys = self.__cos_similarity(identifys, self.identifysDb)
        similaritys[np.isnan(similaritys)] = 0
        ids = np.nanargmax(similaritys, axis=1)

        for i, similarity in enumerate(similaritys):
            personId = ids[i]
            d = self._getDistance(detections[i], personId)
            print("persionId:{} {} conf:{}".format(personId, similarity[personId], detections[i][4]))
            # If 0.9 or higher and the face detection confidence is higher than the existing one, the identification information is updated
            if (similarity[personId] > 0.9 and detections[i][4] > self.conf[personId]):
                print("? refresh id:{} conf:{}".format(personId, detections[i][4]))
                self.identifysDb[personId] = identifys[i]
            # If less than 0.3, add
            elif (similarity[personId] < 0.3):
                if (d > 100):
                    self.identifysDb = np.vstack((self.identifysDb, identifys[i]))
                    self.conf.append(detections[i][4])
                    ids[i] = len(self.identifysDb) - 1
                    print("append id:{} similarity:{}".format(ids[i], similarity[personId]))

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

class FaceEmotionRecognition(Model):

    def __init__(self, path_model):

        self.model_name = "emotions-recognition-retail-0003"

        super().__init__(path_model, self.model_name)

    def infer(self, image, faces):

        emotions = np.zeros((len(faces), 5))

        for i, face in enumerate(faces):

            img = image[face[1]: face[3], face[0]: face[2]]
            h, w = img.shape[:2]

            if (h == 0 or w == 0):
                continue

            scaled_frame, scale_h, scale_w = self.scale_frame(frame=img)

            result = self.compiled_model([scaled_frame])[self.output_layer]
            emotions[i] = result[0, :, 0, 0]

        return emotions

class FaceMesh():

    def __init__(self):

        self._mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh_parameters = {"max_num_faces": 1,
                                     "refine_landmarks": True,
                                     "min_detection_confidence": 0.5,
                                     "min_tracking_confidence": 0.5}

    def process(self, image, person_face_results):

        MARGIN = 10
        results = {}

        for i, face in enumerate(person_face_results):

            xmin = face[0]
            xmax = face[2]
            ymin = face[1]
            ymax = face[3]

            img = image[(int(ymin) - MARGIN):(int(ymax) + MARGIN), (int(xmin) - MARGIN):(int(xmax) + MARGIN)]

            with self._mp_face_mesh.FaceMesh(**self.face_mesh_parameters) as face_mesh:

                results[i] = face_mesh.process(img)

        return results

    @property
    def mp_face_mesh(self):
        return self._mp_face_mesh

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

        self._get_model()
        self._set_results_dict()

    def process(self):

        self._detect_person_faces()

        self._detect_tracks()

        self._detect_face_emotions()

        self._detect_face_mesh()

    def _get_model(self):

        self.person_face_detector = FaceDetector(path_model=self.path_model)

        self.face_re_identification = FaceReidentification(path_model=self.path_model)
        self.tracker = Tracker()

        self.face_emotion_recognition = FaceEmotionRecognition(path_model=self.path_model)

        self._face_mesh = FaceMesh()

    def _set_results_dict(self):

        # Set dictionary for storing person face detection results
        self._person_face_results = {}
        self._detection_results = {}
        self._confidence_results = {}

        # for storing face re-identification results
        self._identifies_results = {}
        self._ids_results = {}

        # For storing face emotion recognition results
        self._emotion_results = {}

        # For storing face mesh results
        self._mp_results = {}

    def _detect_person_faces(self):

        for i in range(self.n_frame):

            image = copy.deepcopy(self.data_frame_dict[i])

            self._detection_results[i], \
            self._person_face_results[i] = self.person_face_detector.infer(image=image)
            self._confidence_results[i] = self.person_face_detector.conf

            print("Processing face BBox at frame: {}/{}".format(i, self.n_frame))

    def _detect_tracks(self):

        for i in range(self.n_frame):

            image = copy.deepcopy(self.data_frame_dict[i])

            self._identifies_results[i] = self.face_re_identification.infer(image=image,
                                                                            person_faces=self._person_face_results[i])

            self._ids_results[i] = self.tracker.getIds(identifys=self._identifies_results[i],
                                                       detections=self._detection_results[i])

            print("Processing ID at frame: {}/{}".format(i, self.n_frame))

    def _detect_face_emotions(self):

        for i in range(self.n_frame):

            image = copy.deepcopy(self.data_frame_dict[i])

            self._emotion_results[i] = self.face_emotion_recognition.infer(image=image,
                                                                           faces=self._person_face_results[i])

            print("Processing face emotion recognition: {}/{}".format(i, self.n_frame))

    def _detect_face_mesh(self):

        for i in range(self.n_frame):

            image = copy.deepcopy(self.data_frame_dict[i])

            self._mp_results[i] = self._face_mesh.process(image=image,
                                                          person_face_results=self._person_face_results[i])

            print("Processing face mesh: {}/{}".format(i, self.n_frame))

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

    @property
    def emotion_results(self):
        return self._emotion_results

    @property
    def mp_results(self):
        return self._mp_results

    @property
    def face_mesh(self):
        return self._face_mesh


plot_parameters = {"scale_image": 3,
                   "scale_text": 2,
                   "thickness": 3,
                   "margin": 10}

class VideoManager():

    def __init__(self, data_frame_dict: dict, plot_parameters: dict):

        self.SCALE_IMAGE = plot_parameters["scale_image"]
        self.SCALE_TEXT = plot_parameters["scale_text"]
        self.THICKNESS = plot_parameters["thickness"]
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

                image = DrawingUtilities.put_highlight_text(image=image,
                                                            text="Bounding Box: OpenVINO",
                                                            org=(0, 30),
                                                            scale_text=self.SCALE_TEXT,
                                                            color=Drawing.color_white,
                                                            thickness=self.THICKNESS)

            image = self._draw_bbox(image=image,
                                    person_faces=person_face_results[i],
                                    conf=conf[i],
                                    color=(255, 255, 255))

            self._image_list.append(image)
            print("Drawing BBox at frame: " + str(i) + "/" + str(self.n_frame))

        save_path = video_save_folder + video_name + "_bbox_face_OpenVINO.mp4"
        VideoUtilities.save_images_to_video(images=self._image_list,
                                            save_path=save_path,
                                            image_height=image_height,
                                            image_width=image_width)

    def write_emotion(self,
                      person_face_results: dict,
                      ids_results: dict,
                      emotion_results: dict,
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

                image = DrawingUtilities.put_highlight_text(image=image,
                                                            text="Bounding Box: OpenVINO",
                                                            org=(0, 30),
                                                            scale_text=self.SCALE_TEXT,
                                                            color=Drawing.color_white,
                                                            thickness=self.THICKNESS)

                image = DrawingUtilities.put_highlight_text(image=image,
                                                            text="Tracks (Ids): OpenVINO",
                                                            org=(0, 60),
                                                            scale_text=self.SCALE_TEXT,
                                                            color=Drawing.color_white,
                                                            thickness=self.THICKNESS)

                image = DrawingUtilities.put_highlight_text(image=image,
                                                            text="Emotion recognition: OpenVINO",
                                                            org=(0, 90),
                                                            scale_text=self.SCALE_TEXT,
                                                            color=Drawing.color_white,
                                                            thickness=self.THICKNESS)

            image = self._draw_emotions(image=image,
                                        faces=person_face_results[i],
                                        ids_results=ids_results[i],
                                        emotion_results=emotion_results[i],
                                        colors=colors)

            self._image_list.append(image)
            print("Drawing BBox and Emotions at frame: " + str(i) + "/" + str(self.n_frame))

        save_path = video_save_folder + video_name + "_bbox_emotion_face_OpenVINO.mp4"
        VideoUtilities.save_images_to_video(images=self._image_list,
                                            save_path=save_path,
                                            image_height=image_height,
                                            image_width=image_width)

    def write_face_mesh(self,
                        person_face_results: dict,
                        ids_results: dict,
                        emotion_results: dict,
                        mp_results: dict,
                        face_mesh: FaceMesh,
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

                image = DrawingUtilities.put_highlight_text(image=image,
                                                            text="Bounding Box: OpenVINO",
                                                            org=(0, 30),
                                                            scale_text=self.SCALE_TEXT,
                                                            color=Drawing.color_white,
                                                            thickness=self.THICKNESS)

                image = DrawingUtilities.put_highlight_text(image=image,
                                                            text="Tracks (Ids): OpenVINO",
                                                            org=(0, 60),
                                                            scale_text=self.SCALE_TEXT,
                                                            color=Drawing.color_white,
                                                            thickness=self.THICKNESS)

                image = DrawingUtilities.put_highlight_text(image=image,
                                                            text="Emotion recognition: OpenVINO",
                                                            org=(0, 90),
                                                            scale_text=self.SCALE_TEXT,
                                                            color=Drawing.color_white,
                                                            thickness=self.THICKNESS)

                image = DrawingUtilities.put_highlight_text(image=image,
                                                            text="Face mesh: MediaPipe",
                                                            org=(0, 120),
                                                            scale_text=self.SCALE_TEXT,
                                                            color=Drawing.color_white,
                                                            thickness=self.THICKNESS)

            image = self._draw_emotions(image=image,
                                        faces=person_face_results[i],
                                        ids_results=ids_results[i],
                                        emotion_results=emotion_results[i],
                                        colors=colors)

            image = self._draw_face_mesh(image=image,
                                         faces=person_face_results[i],
                                         ids_results=ids_results[i],
                                         colors=colors,
                                         face_mesh=face_mesh,
                                         mp_results=mp_results[i])

            self._image_list.append(image)
            print("Drawing BBox and Emotions at frame: " + str(i) + "/" + str(self.n_frame))

        save_path = video_save_folder + video_name + "_bbox_emotion_facemesh_OpenVINO.mp4"
        VideoUtilities.save_images_to_video(images=self._image_list,
                                            save_path=save_path,
                                            image_height=image_height,
                                            image_width=image_width)

    def _draw_bbox(self, image, person_faces, conf, color):

        for f, face in enumerate(person_faces):

            start_point = (face[0] - self.MARGIN, face[1] - self.MARGIN)
            end_point = (face[2] + self.MARGIN, face[3] + self.MARGIN)

            image = DrawingUtilities.put_highlight_rectangle(image=image,
                                                             pt1=start_point,
                                                             pt2=end_point,
                                                             color=color,
                                                             thickness=self.THICKNESS)

            image = DrawingUtilities.put_highlight_text(image=image,
                                                        text="confidence: {:.2f}".format(conf[f]),
                                                        org=(face[0] - self.MARGIN, face[1] - self.MARGIN * 2),
                                                        scale_text=self.SCALE_TEXT,
                                                        color=color,
                                                        thickness=self.THICKNESS)

        return image

    def _draw_emotions(self, image, faces, ids_results, emotion_results, colors):

        for i, face in enumerate(faces):

            color = colors[int(ids_results[i])]

            start_point = (face[0] - self.MARGIN, face[1] - self.MARGIN)
            end_point = (face[2] + self.MARGIN, face[3] + self.MARGIN)

            image = DrawingUtilities.put_highlight_rectangle(image=image,
                                                             pt1=start_point,
                                                             pt2=end_point,
                                                             color=color,
                                                             thickness=self.THICKNESS)

            label = ('neutral', 'happy', 'sad', 'surprise', 'anger')

            emotion = label[np.argmax(emotion_results[i].squeeze())]
            prob = np.max(emotion_results[i].squeeze())

            image = DrawingUtilities.put_highlight_text(image=image,
                                                        text="{}: {} - {:.1f}".format(ids_results[i], emotion, prob),
                                                        org=(face[0] - self.MARGIN * 2, face[1] - self.MARGIN * 2),
                                                        scale_text=self.SCALE_TEXT,
                                                        color=color,
                                                        thickness=self.THICKNESS)

        return image

    def _draw_face_mesh(self, image, faces, ids_results, colors, face_mesh: FaceMesh, mp_results: dict):

        for i, face in enumerate(faces):

            color = colors[int(ids_results[i])]
            color_inv = (255 - color[0], 255 - color[1], 255 - color[2])

            # Face mesh

            img = image[(int(face[1]) - self.MARGIN):(int(face[3]) + self.MARGIN),
                  (int(face[0]) - self.MARGIN):(int(face[2]) + self.MARGIN)]

            if mp_results[i].multi_face_landmarks:

                for face_landmarks in mp_results[i].multi_face_landmarks:

                    # Drawing.mp_drawing.draw_landmarks(image=img,
                    #                                   landmark_list=face_landmarks,
                    #                                   connections=face_mesh.mp_face_mesh.FACEMESH_TESSELATION,
                    #                                   landmark_drawing_spec=Drawing.mp_drawing.DrawingSpec(color=color, thickness=1, circle_radius=2),
                    #                                   connection_drawing_spec=Drawing.mp_drawing.DrawingSpec(color=Drawing.color_white, thickness=1))

                    Drawing.mp_drawing.draw_landmarks(image=img,
                                                      landmark_list=face_landmarks,
                                                      connections=face_mesh.mp_face_mesh.FACEMESH_CONTOURS,
                                                      landmark_drawing_spec=Drawing.mp_drawing.DrawingSpec(color=color, thickness=1, circle_radius=3),
                                                      connection_drawing_spec=Drawing.mp_drawing.DrawingSpec(color=Drawing.color_white, thickness=1))

                    Drawing.mp_drawing.draw_landmarks(image=img,
                                                      landmark_list=face_landmarks,
                                                      connections=face_mesh.mp_face_mesh.FACEMESH_IRISES,
                                                      landmark_drawing_spec=None,
                                                      connection_drawing_spec=Drawing.mp_drawing.DrawingSpec(color=color_inv, thickness=3))

        return image

pathInfo = PathInfo()
video_folder = pathInfo.path_data_test
video_name = "Interview_Video_0"
video_format = ".mp4"

cap = cv2.VideoCapture(video_folder + video_name + video_format)
processVideo = ProcessVideo(video_path=video_folder + video_name + video_format)
data_frame_dict = processVideo.read(n_frame=500, resize_param=plot_parameters["scale_image"])

manager = Manager(data_frame_dict=data_frame_dict,
                  path_model=pathInfo.path_model)
manager.process()

emotions = manager.emotion_results

videoManager = VideoManager(data_frame_dict=data_frame_dict,
                            plot_parameters=plot_parameters)

videoManager.write_bbox(person_face_results=manager.person_face_results,
                        conf=manager.confidence_results,
                        video_save_folder=KentaPathInfo().saved_video_path,
                        video_name=video_name)

videoManager.write_emotion(person_face_results=manager.person_face_results,
                           ids_results=manager.ids_results,
                           emotion_results=manager.emotion_results,
                           video_save_folder=KentaPathInfo().saved_video_path,
                           video_name=video_name,
                           is_info=False)

videoManager.write_face_mesh(person_face_results=manager.person_face_results,
                             ids_results=manager.ids_results,
                             emotion_results=manager.emotion_results,
                             mp_results=manager.mp_results,
                             face_mesh=manager.face_mesh,
                             video_save_folder=KentaPathInfo().saved_video_path,
                             video_name=video_name,
                             is_info=False)


