from retinaface import RetinaFace
import copy


class PersonFaceDetectorRF():

    def __init__(self, path_model: str = "", threshold:float = 0.6):

        self.path_model = path_model
        self.model_name = "retina_face"
        self.threshold = threshold

        # initialize the list of confidence for each face
        self._conf = []

    def infer(self, image):

        img = copy.deepcopy(image)

        detections = self._get_detection(image=img)

        persons = self._get_persons(detections=detections)

        return detections, persons

    def _get_persons(self, detections):

        persons = []
        self._conf = []

        if detections:
            print("--------- faces: {} ---------".format(len(detections)))
            for key in detections:

                identify = detections[key]
                facial_area = identify["facial_area"]
                conf = identify["score"]

                # the facial are is already int type
                box_xmin = int(facial_area[0])
                box_ymin = int(facial_area[1])
                box_xmax = int(facial_area[2])
                box_ymax = int(facial_area[3])

                box = [box_xmin, box_ymin, box_xmax, box_ymax]
                print("Conf - {:.1f}, Location - ({}, {})-({}, {})".format(conf, box_xmin, box_ymin, box_xmax, box_ymax))
                self._conf.append(float(conf))
                persons.append(box)

        else:
            self._conf = [None]

        return persons

    def _get_detection(self, image):

        detections = RetinaFace.detect_faces(image, threshold=self.threshold)

        return detections

    @property
    def conf(self):
        return self._conf