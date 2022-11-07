import numpy as np
import time
import random
import cv2
from openvino.runtime import Core
from openvino.inference_engine import IECore
from Classes.PathInfo import PathInfo
from Classes.Video import *

class Model:

    def __init__(self, model_path, device, ie_core, num_requests, output_shape=None):

        if model_path.endswith((".xml", ".bin")):
            model_path = model_path[:-4]
        self.net = ie_core.read_network(model_path + ".xml", model_path + ".bin")
        assert len(self.net.input_info) == 1, "One input is expected"

        # supported_layers = ie_core.query_network(self.net, device)
        # not_supported_layers = [l for l in self.net.layers.keys() if l not in supported_layers]
        # if len(not_supported_layers) > 0:
        #     raise RuntimeError("Following layers are not supported by the {} plugin:\n {}"
        #                        .format(device, ', '.join(not_supported_layers)))

        self.exec_net = ie_core.load_network(network=self.net,
                                             device_name=device,
                                             num_requests=num_requests)

        self.input_name = next(iter(self.net.input_info))
        if len(self.net.outputs) > 1:
            if output_shape is not None:
                candidates = []
                for candidate_name in self.net.outputs:
                    candidate_shape = self.exec_net.requests[0].output_blobs[candidate_name].buffer.shape
                    if len(candidate_shape) != len(output_shape):
                        continue

                    matches = [src == trg or trg < 0
                               for src, trg in zip(candidate_shape, output_shape)]
                    if all(matches):
                        candidates.append(candidate_name)

                if len(candidates) != 1:
                    raise Exception("One output is expected")

                self.output_name = candidates[0]
            else:
                raise Exception("One output is expected")
        else:
            self.output_name = next(iter(self.net.outputs))

        self.input_size = self.net.input_info[self.input_name].input_data.shape
        self.output_size = self.exec_net.requests[0].output_blobs[self.output_name].buffer.shape
        self.num_requests = num_requests

    def infer(self, data):
        input_data = {self.input_name: data}
        infer_result = self.exec_net.infer(input_data)
        return infer_result[self.output_name]


class PersonDetector(Model):

    def __init__(self, model_path, device, ie_core, threshold, num_requests):

        super().__init__(model_path, device, ie_core, num_requests, None)
        _, _, h, w = self.input_size
        self.__input_height = h
        self.__input_width = w
        self.__threshold = threshold

    def __prepare_frame(self, frame):

        initial_h, initial_w = frame.shape[:2]
        scale_h, scale_w = initial_h / float(self.__input_height), initial_w / float(self.__input_width)
        in_frame = cv2.resize(frame, (self.__input_width, self.__input_height))
        in_frame = in_frame.transpose((2, 0, 1))
        in_frame = in_frame.reshape(self.input_size)

        return in_frame, scale_h, scale_w

    def infer(self, frame):

        in_frame, _, _ = self.__prepare_frame(frame)
        result = super().infer(in_frame)

        detections = []
        height, width = frame.shape[:2]
        for r in result[0][0]:
            conf = r[2]
            if (conf > self.__threshold):
                x1 = int(r[3] * width)
                y1 = int(r[4] * height)
                x2 = int(r[5] * width)
                y2 = int(r[6] * height)
                detections.append([x1, y1, x2, y2, conf])

        return detections


class PersonReidentification(Model):

    def __init__(self, model_path, device, ie_core, threshold, num_requests):
        super().__init__(model_path, device, ie_core, num_requests, None)
        _, _, h, w = self.input_size
        self.__input_height = h
        self.__input_width = w
        self.__threshold = threshold

    def __prepare_frame(self, frame):
        initial_h, initial_w = frame.shape[:2]
        scale_h, scale_w = initial_h / float(self.__input_height), initial_w / float(self.__input_width)
        in_frame = cv2.resize(frame, (self.__input_width, self.__input_height))
        in_frame = in_frame.transpose((2, 0, 1))
        in_frame = in_frame.reshape(self.input_size)
        return in_frame, scale_h, scale_w

    def infer(self, frame):
        in_frame, _, _ = self.__prepare_frame(frame)
        result = super().infer(in_frame)
        return np.delete(result, 1)


class Tracker:

    def __init__(self):

        # Database of identification information
        self.identifiesDb = None
        # Database of center locations
        self.center = []

    def __getCenter(self, person):
        x = person[0] - person[2]
        y = person[1] - person[3]
        return (x, y)

    def __getDistance(self, person, index):
        (x1, y1) = self.center[index]
        (x2, y2) = self.__getCenter(person)
        a = np.array([x1, y1])
        b = np.array([x2, y2])
        u = b - a
        return np.linalg.norm(u)

    def __isOverlap(self, persons, index):
        [x1, y1, x2, y2] = persons[index]
        for i, person in enumerate(persons):
            if (index == i):
                continue
            if (max(person[0], x1) <= min(person[2], x2) and max(person[1], y1) <= min(person[3], y2)):
                return True
        return False

    def getIds(self, identifies, persons):
        if (identifies.size == 0):
            return []
        if self.identifiesDb is None:
            self.identifiesDb = identifies
            for person in persons:
                self.center.append(self.__getCenter(person))

        print("input: {} DB:{}".format(len(identifies), len(self.identifiesDb)))
        similaritys = self.__cos_similarity(identifies, self.identifiesDb)
        similaritys[np.isnan(similaritys)] = 0
        ids = np.nanargmax(similaritys, axis=1)

        for i, similarity in enumerate(similaritys):
            persionId = ids[i]
            d = self.__getDistance(persons[i], persionId)
            print("persionId:{} {} distance:{}".format(persionId, similarity[persionId], d))
            # If similarity is greater than 0.95 and there is no overlap, the identification information is updated
            if (similarity[persionId] > 0.95):
                if (self.__isOverlap(persons, i) == False):
                    self.identifiesDb[persionId] = identifies[i]
            # If similarity is less than 0.5 and the distance is far, register a new ID
            elif (similarity[persionId] < 0.5):
                if (d > 500):
                    print("distance:{} similarity:{}".format(d, similarity[persionId]))
                    self.identifiesDb = np.vstack((self.identifiesDb, identifies[i]))
                    self.center.append(self.__getCenter(persons[i]))
                    ids[i] = len(self.identifiesDb) - 1
                    print("> append DB size:{}".format(len(self.identifiesDb)))

        print(ids)
        # If there are duplicate IDs, the one with the lower confidence level is invalidated.
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


device = "CPU"
cpu_extension = None
ie_core = IECore()
if device == "CPU" and cpu_extension:
    ie_core.add_extension(cpu_extension, "CPU")

THRESHOLD = 0.8

path_info = PathInfo()

path_person_detection = path_info.path_model + "\\person-detection-retail-0013"
path_person_reidentification = path_info.path_model + "\\person-reidentification-retail-0288"
video_path = path_info.path_data_test + "Video_test.mp4"
# video_path = path_info.path_data_test + "Video_test_2.mp4"

person_detector = PersonDetector(path_person_detection, device, ie_core, THRESHOLD, num_requests=2)

personReidentification = PersonReidentification(path_person_reidentification, device, ie_core, THRESHOLD,
                                                num_requests=2)
tracker = Tracker()

SCALE_IMAGE = 1
SCALE_ID = 0.3

cap = cv2.VideoCapture(video_path)

TRACKING_MAX = 50
colors = []
for i in range(TRACKING_MAX):
    b = random.randint(0, 255)
    g = random.randint(0, 255)
    r = random.randint(0, 255)
    colors.append((b, g, r))

while True:

    grabbed, frame = cap.read()
    if not grabbed:  # loop playback
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    if (frame is None):
        continue

    # Person Detection
    persons = []
    detections = person_detector.infer(frame)
    if (len(detections) > 0):
        print("-------------------")
        for detection in detections:
            x1 = int(detection[0])
            y1 = int(detection[1])
            x2 = int(detection[2])
            y2 = int(detection[3])
            conf = detection[4]
            print("{:.1f} ({},{})-({},{})".format(conf, x1, y1, x2, y2))
            persons.append([x1, y1, x2, y2])

    # [[172, 146, 300, 561], [549, 88, 729, 639], [972, 139, 1103, 572]]

    print("====================")
    # Obtain identification information from each Person's image
    identifies = np.zeros((len(persons), 255))
    for i, person in enumerate(persons):
        # Acquisition of each person's image
        img = frame[person[1]: person[3], person[0]: person[2]]
        h, w = img.shape[:2]
        if (h == 0 or w == 0):
            continue
        # Identification information acquisition
        identifies[i] = personReidentification.infer(img)

    # 6.67167783e-01,  3.77045155e-01, -1.74978860e-02,
    #         -8.52454424e-01, -3.39285791e-01, -8.53898898e-02

    # Get Ids
    ids = tracker.getIds(identifies, persons)

    #  array([0, 1, 2], dtype=int64)

    # Add frame and ID to image
    for i, person in enumerate(persons):
        if (ids[i] != -1):
            color = colors[int(ids[i])]
            frame = cv2.rectangle(frame, (person[0], person[1]), (person[2], person[3]), color, int(50 * SCALE_ID))
            frame = cv2.putText(frame, str(ids[i]), (person[0], person[1]), cv2.FONT_HERSHEY_PLAIN, int(50 * SCALE_ID),
                                color, int(30 * SCALE_ID), cv2.LINE_AA)

    # You can reduce the size of an image by changing SCALE_IMAGE.
    h, w = frame.shape[:2]
    frame = cv2.resize(frame, ((int(w * SCALE_IMAGE), int(h * SCALE_IMAGE))))
    # Appear Image
    cv2.imshow('Person Detection and Reidentification', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()