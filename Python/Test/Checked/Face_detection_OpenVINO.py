from ClassesML.FaceDetectionModel import FaceDetectionModel
from Classes.PathInfo import PathInfo
from Classes.Video import ProcessVideo, VideoUtilities
from Kenta.TempClasses.PathInfo import KentaPathInfo
import cv2
import copy


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

class Manager():

    def __init__(self, data_frame_dict, path_model):

        self.data_frame_dict = data_frame_dict
        self.path_model = path_model

        self.n_frame = len(self.data_frame_dict.keys())

        self._get_model()
        self._set_results_dict()

    def process(self):

        self._detect_person_faces()

    def _get_model(self):

        self.person_face_detector = FaceDetector(path_model=self.path_model)

    def _set_results_dict(self):

        # Set dictionary for storing person face detection results
        self._person_face_results = {}
        self._detection_results = {}
        self._confidence_results = {}

    def _detect_person_faces(self):

        for i in range(self.n_frame):

            image = copy.deepcopy(self.data_frame_dict[i])

            self._detection_results[i], \
            self._person_face_results[i] = self.person_face_detector.infer(image=image)
            self._confidence_results[i] = self.person_face_detector.conf

            print("Processing face BBox at frame: " + str(i) + "/" + str(self.n_frame))

    @property
    def person_face_results(self):
        return self._person_face_results

    @property
    def detection_results(self):
        return self._detection_results

    @property
    def confidence_results(self):
        return self._confidence_results

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

    def write_bbox(self, person_face_results: dict, conf: dict, video_save_folder: str, video_name: str):

        self._image_list = []
        image_height = 0
        image_width = 0

        for i in range(self.n_frame):

            image = copy.deepcopy(self.data_frame_dict[i])
            image_height, image_width, _ = image.shape

            image = self._draw_bbox(image=image,
                                    person_faces=person_face_results[i],
                                    conf=conf[i],
                                    color=(255, 0, 0))



            self._image_list.append(image)
            print("Drawing BBox at frame: " + str(i) + "/" + str(self.n_frame))

        save_path = video_save_folder + video_name + "bbox_face_OpenVINO.mp4"
        VideoUtilities.save_images_to_video(images=self._image_list,
                                            save_path=save_path,
                                            image_height=image_height,
                                            image_width=image_width)

    def _draw_bbox(self, image, person_faces, conf, color):

        SCALE_TEXT = 3

        image = cv2.putText(image, "Bounding Box: OpenVINO",
                            (0, 50),
                            cv2.FONT_HERSHEY_PLAIN,
                            SCALE_TEXT,
                            (255, 0, 0),
                            SCALE_TEXT,
                            cv2.LINE_AA)

        for f, face in enumerate(person_faces):

            start_point = (face[0] - self.MARGIN, face[1] - self.MARGIN)
            end_point = (face[2] + self.MARGIN, face[3] + self.MARGIN)

            image = cv2.rectangle(image,
                                  pt1=start_point,
                                  pt2=end_point,
                                  color=color,
                                  thickness=2)

            image = cv2.putText(image,
                                text="confidence: {:.2f}".format(conf[f]),
                                org=start_point,
                                fontFace=cv2.FONT_HERSHEY_PLAIN,
                                fontScale=self.SCALE_TEXT,
                                color=(0, 255, 0),
                                lineType=cv2.LINE_AA)

        return image


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

videoManager.write_bbox(person_face_results=manager.person_face_results,
                        conf=manager.confidence_results,
                        video_save_folder=KentaPathInfo().saved_video_path,
                        video_name=video_name)
