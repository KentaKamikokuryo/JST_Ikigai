from Classes.PathInfo import PathInfo
from Classes.Drawing import Drawing, DrawingUtilities
from Classes.Video import ProcessVideo, VideoUtilities
from ClassesML.FaceDetectionModel import FaceDetectionModel
from Kenta.TempClasses.PathInfo import KentaPathInfo
import cv2
import copy
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


class FaceDetectionOV(FaceDetectionModel):

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

    @property
    def conf(self):
        return self._conf

class FaceDetectionMediaPipe():

    def __init__(self):

        self.mp_face_detection = mp.solutions.face_detection
        self._confs = []

        """
        FACE_DETECTION_PARAMETERS: 
        model_selection: This argument takes the real integer value only in the range of 0-1 i.e. this model will take the integer value as either 1 or 0. Let us discuss these two types of models.
            - 0 type model: When we will select the 0 type model then our face detection model will be able to detect the faces within the range of 2 meters from the camera.
            - 1 type model: When we will select the 1 type model then our face detection model will be able to detect the faces within the range of 5 meters. Though the default value is 0.
        min_detection_confidence: This argument also takes the integer value but in the range of [0.0,1.0] and the default value for the same is 0.5 which is 50% confidence i.e. when our model will be detecting the faces it should be at least 50% sure that the face is there otherwise it won’t detect anything.
        """
        self.face_detection_parameters = dict(model_selection=1,
                                              min_detection_confidence=0.4)

    def infer(self, image):

        img = copy.deepcopy(image)

        img, detections = self._get_detection(image=img)

        persons = self._get_persons(detections=detections)

        return detections, persons

    def _get_persons(self, detections):

        persons = []
        if (len(detections) > 0):
            print("-------------------")
            for n in range(len(detections)):

                detection = detections[n]
                temp = detection.location_data.relative_bounding_box
                conf = detection.score[0]

                box_xmin = int(temp.xmin * self.image_width)
                box_ymin = int(temp.ymin * self.image_height)
                box_xmax = int(box_xmin + temp.width * self.image_width)
                box_ymax = int(box_ymin + temp.height * self.image_height)
                box = [box_xmin, box_ymin, box_xmax, box_ymax]

                print("Conf - {:.1f}, Location - ({}, {})-({}, {})".format(conf, box_xmin, box_ymin, box_xmax, box_ymax))
                self._confs.append(conf)
                persons.append(box)

        return persons

    def _get_detection(self, image):

        with self.mp_face_detection.FaceDetection(**self.face_detection_parameters) as face_detection:
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.image_height, self.image_width, _ = image.shape

            results_temp = face_detection.process(image)
            detections = results_temp.detections

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        return image, detections

    @property
    def conf(self):
        return self._confs

class Manager():

    def __init__(self, data_frame_dict, path_model):

        self.data_frame_dict = data_frame_dict
        self.path_model = path_model

        self.n_frame = len(self.data_frame_dict.keys())

        self._get_model()
        self._set_results_dict()

    def process(self):

        self._detect_person_faces_ov()
        self._detect_person_faces_mp()

    def _get_model(self):

        self.person_face_detector_OV = FaceDetectionOV(path_model=self.path_model)
        self.person_face_detector_MP = FaceDetectionMediaPipe()

    def _set_results_dict(self):

        # Set dictionary for storing person face detection results
        self._person_face_results_OV = {}
        self._detection_results_OV = {}
        self._confidence_results_OV = {}

        self._person_face_results_MP = {}
        self._detection_results_MP = {}
        self._confidence_results_MP = {}

    def _detect_person_faces_ov(self):

        for i in range(self.n_frame):

            image = copy.deepcopy(self.data_frame_dict[i])

            self._detection_results_OV[i], \
            self._person_face_results_OV[i] = self.person_face_detector_OV.infer(image=image)
            self._confidence_results_OV[i] = self.person_face_detector_OV.conf

            print("OpenVINO - Processing face BBox at frame: " + str(i) + "/" + str(self.n_frame))

    def _detect_person_faces_mp(self):

        for i in range(self.n_frame):

            image = copy.deepcopy(self.data_frame_dict[i])

            self._detection_results_MP[i], \
            self._person_face_results_MP[i] = self.person_face_detector_MP.infer(image=image)
            self._confidence_results_MP[i] = self.person_face_detector_MP.conf

            print("MediaPipe - Processing face BBox at frame: " + str(i) + "/" + str(self.n_frame))

    @property
    def person_face_results_ov(self):
        return self._person_face_results_OV

    @property
    def detection_results_ov(self):
        return self._detection_results_OV

    @property
    def confidence_results_ov(self):
        return self._confidence_results_OV

    @property
    def person_face_results_mp(self):
        return self._person_face_results_MP

    @property
    def detection_results_mp(self):
        return self._detection_results_MP

    @property
    def confidence_results_mp(self):
        return self._confidence_results_MP


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
                   person_face_results_ov: dict,
                   person_face_results_mp: dict,
                   conf_ov: dict,
                   conf_mp: dict,
                   video_save_folder: str, video_name: str):

        self._image_list = []
        image_height = 0
        image_width = 0
        SCALE_TEXT = 2

        for i in range(self.n_frame):

            image = copy.deepcopy(self.data_frame_dict[i])
            image_height, image_width, _ = image.shape

            SCALE_TEXT = 3

            image = DrawingUtilities.put_highlight_text(image=image,
                                                        text="Bounding Box: OpenVINO",
                                                        org=(10, 50),
                                                        scale_text=SCALE_TEXT,
                                                        color=Drawing.color_blue,
                                                        thickness=2)

            image = DrawingUtilities.put_highlight_text(image=image,
                                                        text="Bounding Box: MediaPipe",
                                                        org=(10, 100),
                                                        scale_text=SCALE_TEXT,
                                                        color=Drawing.color_red,
                                                        thickness=2)

            image = self._draw_bbox(image=image,
                                    person_faces=person_face_results_ov[i],
                                    conf=conf_ov[i],
                                    color=Drawing.color_blue,
                                    org_pos=2)

            image = self._draw_bbox(image=image,
                                    person_faces=person_face_results_mp[i],
                                    conf=conf_mp[i],
                                    color=Drawing.color_red,
                                    org_pos=4)

            self._image_list.append(image)
            print("Drawing BBox at frame: " + str(i) + "/" + str(self.n_frame))

        save_path = video_save_folder + video_name + "_bbox_face_Comparison.mp4"
        VideoUtilities.save_images_to_video(images=self._image_list,
                                            save_path=save_path,
                                            image_height=image_height,
                                            image_width=image_width)

    def _draw_bbox(self, image, person_faces, conf, color, org_pos: int):

        for f, face in enumerate(person_faces):

            start_point = (face[0] - self.MARGIN, face[1] - self.MARGIN)
            end_point = (face[2] + self.MARGIN, face[3] + self.MARGIN)

            image = DrawingUtilities.put_highlight_rectangle(image=image,
                                                             pt1=start_point,
                                                             pt2=end_point,
                                                             color=color,
                                                             thickness=3)

            image = DrawingUtilities.put_highlight_text(image=image,
                                                        text="conf: {:.2f}".format(conf[f]),
                                                        org=(face[0] - self.MARGIN, face[1] - self.MARGIN * org_pos),
                                                        scale_text=2,
                                                        color=color,
                                                        thickness=2)

        return image


pathInfo = PathInfo()
video_folder = pathInfo.path_data_test
video_name = "Interview_Video_0"
video_format = ".mp4"

cap = cv2.VideoCapture(video_folder + video_name + video_format)
processVideo = ProcessVideo(video_path=video_folder + video_name + video_format)
data_frame_dict = processVideo.read(n_frame=200, resize_param=3)

manager = Manager(data_frame_dict=data_frame_dict,
                  path_model=pathInfo.path_model)
self = manager
manager.process()

videoManager = VideoManager(data_frame_dict=data_frame_dict,
                            plot_parameters=plot_parameters)

videoManager.write_bbox(person_face_results_ov=manager.person_face_results_ov,
                        conf_ov=manager.confidence_results_ov,
                        person_face_results_mp=manager.person_face_results_mp,
                        conf_mp=manager.confidence_results_mp,
                        video_save_folder=KentaPathInfo().saved_video_path,
                        video_name=video_name)
