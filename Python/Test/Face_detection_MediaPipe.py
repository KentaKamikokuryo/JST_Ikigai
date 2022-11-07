from Classes.PathInfo import PathInfo
from Classes.Video import ProcessVideo, VideoUtilities
from Kenta.TempClasses.PathInfo import KentaPathInfo
import cv2
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
import copy

# # For static images:
# IMAGE_FILES = []
# with mp_face_detection.FaceDetection(
#     model_selection=1, min_detection_confidence=0.5) as face_detection:
#   for idx, file in enumerate(IMAGE_FILES):
#     image = cv2.imread(file)
#     # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
#     results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#
#     # Draw face detections of each face.
#     if not results.detections:
#       continue
#     annotated_image = image.copy()
#     for detection in results.detections:
#       print('Nose tip:')
#       print(mp_face_detection.get_key_point(
#           detection, mp_face_detection.FaceKeyPoint.NOSE_TIP))
#       mp_drawing.draw_detection(annotated_image, detection)
#     cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)

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

        return persons, detections

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

    def __init__(self, data_frame_dict: dict, path_model: str):

        self.data_frame_dict = data_frame_dict
        self.path_model = path_model

        self.n_frame = len(self.data_frame_dict.keys())

        self._get_model()
        self._set_results_dict()

    def process(self):

        self._detect_person_faces()

    def _get_model(self):

        self.person_face_detector = FaceDetectionMediaPipe()

    def _set_results_dict(self):

        self._person_results = {}
        self._detection_results = {}

        self._confidence_results = {}

    def _detect_person_faces(self):

        for i in range(self.n_frame):

            img = copy.deepcopy(self.data_frame_dict[i])

            self._person_results[i], \
            self._detection_results[i] = self.person_face_detector.infer(image=img)
            self._confidence_results[i] = self.person_face_detector.conf

            print("Processing face BBox at frame: {}/{}".format(i, self.n_frame))

    @property
    def person_results(self):
        return self._person_results

    @property
    def detection_results(self):
        return self._detection_results

    @property
    def confidence_results(self):
        return self._confidence_results

class VideoManager():

    def __init__(self, data_frame_dict: dict, plot_parameters: dict):

        self.SCALE_IMAGE = plot_parameters["scale_image"]
        self.SCALE_TEXT = plot_parameters["scale_text"]
        self.MARGIN = plot_parameters["margin"]

        self.data_frame_dict = data_frame_dict
        self.n_frame = len(self.data_frame_dict.keys())

        self._image_list = []

    def write_bbox(self, person_face_results: dict, conf: dict, video_save_folder: str, video_name: str):

        self._image_list = []
        image_height = 0
        image_width = 0

        for i in range(self.n_frame):

            image = copy.deepcopy(self.data_frame_dict[i])
            image_height, image_width, _ = image.shape

            image = self._draw_bbox(image=image,
                                    person_faces=person_face_results[i],
                                    conf=conf[i])


            self._image_list.append(image)
            print("Drawing BBox at frame: " + str(i) + "/" + str(self.n_frame))

        save_path = video_save_folder + video_name + "bbox_face_MediaPipe.mp4"
        VideoUtilities.save_images_to_video(images=self._image_list,
                                            save_path=save_path,
                                            image_height=image_height,
                                            image_width=image_width)

    def _draw_bbox(self, image, person_faces, conf):

        SCALE_TEXT = 3

        image = cv2.putText(img=image,
                            text="Bounding Box: MediaPipe",
                            org=(0, 50),
                            fontFace=cv2.FONT_HERSHEY_PLAIN,
                            fontScale=SCALE_TEXT,
                            color=(0, 0, 0),
                            lineType=SCALE_TEXT,
                            thickness=3)

        image = cv2.putText(img=image,
                            text="Bounding Box: MediaPipe",
                            org=(0, 50),
                            fontFace=cv2.FONT_HERSHEY_PLAIN,
                            fontScale=SCALE_TEXT,
                            color=(255, 255, 255),
                            lineType=SCALE_TEXT,
                            thickness=2)

        for f, face in enumerate(person_faces):

            start_point = (face[0] - self.MARGIN, face[1] - self.MARGIN)
            end_point = (face[2] + self.MARGIN, face[3] + self.MARGIN)

            image = cv2.rectangle(image,
                                  pt1=start_point,
                                  pt2=end_point,
                                  color=(0, 0, 0),
                                  thickness=3)

            image = cv2.rectangle(image,
                                  pt1=start_point,
                                  pt2=end_point,
                                  color=(255, 255, 255),
                                  thickness=2)

            image = cv2.putText(image,
                                text="confidence: {:.2f}".format(conf[f]),
                                org=start_point,
                                fontFace=cv2.FONT_HERSHEY_PLAIN,
                                fontScale=self.SCALE_TEXT,
                                color=(0, 0, 0),
                                lineType=cv2.LINE_AA,
                                thickness=2)

            image = cv2.putText(image,
                                text="confidence: {:.2f}".format(conf[f]),
                                org=start_point,
                                fontFace=cv2.FONT_HERSHEY_PLAIN,
                                fontScale=self.SCALE_TEXT,
                                color=(255, 255, 255),
                                lineType=cv2.LINE_AA,
                                thickness=2)

        return image

plot_parameters = {"scale_image": 1,
                   "scale_text": 1.5,
                   "margin": 10}

pathInfo = PathInfo()
video_folder = pathInfo.path_data_test
video_name = "Video_test_0"
video_format = ".mp4"

cap = cv2.VideoCapture(video_folder + video_name + video_format)
processVideo = ProcessVideo(video_path=video_folder + video_name + video_format)
data_frame_dict = processVideo.read(n_frame=70)

manager = Manager(data_frame_dict=data_frame_dict,
                  path_model=pathInfo.path_model)
self = manager
manager.process()

videoManager = VideoManager(data_frame_dict=data_frame_dict,
                            plot_parameters=plot_parameters)

videoManager.write_bbox(person_face_results=manager.person_results,
                        conf=manager.confidence_results,
                        video_save_folder=KentaPathInfo().saved_video_path,
                        video_name=video_name)

# # For webcam input:
# cap = cv2.VideoCapture(0)
# with mp_face_detection.FaceDetection(
#     model_selection=1, min_detection_confidence=0.4) as face_detection:
#   while cap.isOpened():
#     success, image = cap.read()
#     if not success:
#       print("Ignoring empty camera frame.")
#       # If loading a video, use 'break' instead of 'continue'.
#       continue
#
#     # To improve performance, optionally mark the image as not writeable to
#     # pass by reference.
#     image.flags.writeable = False
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = face_detection.process(image)
#
#     # Draw the face detection annotations on the image.
#     image.flags.writeable = True
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     if results.detections:
#       for detection in results.detections:
#         mp_drawing.draw_detection(image, detection)
#     # Flip the image horizontally for a selfie-view display.
#     cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))
#     if cv2.waitKey(5) & 0xFF == 27:
#       break
# cap.release()