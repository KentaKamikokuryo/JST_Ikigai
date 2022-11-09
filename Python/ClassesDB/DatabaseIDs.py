import matplotlib
matplotlib.use('Qt5Agg')
import codecs, json
import os
import matplotlib.pyplot as plt
from Classes.PathInfo import PathInfo
import numpy as np
import cv2
import shutil
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

class ID:

    identify: list
    number: int
    name: str

    def __init__(self, s=None):

        if s is None:
            pass
        else:
            self.__dict__ = json.loads(s)

    def _to_json(self):

        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def save(self, path_folder: str):

        s = self._to_json()

        with open(path_folder + str(self.number) + ".json", "w") as outfile:
            outfile.write(s)

class IDUtilities:

    @staticmethod
    def get_center(person_box):
        x = (person_box[2] + person_box[0]) / 2
        y = (person_box[3] + person_box[1]) / 2
        return (x, y)

    @staticmethod
    def get_distance(person, persons_center, index):

        # person: current box of the person with current index
        # persons_center: center of the box of all person at the previous frame

        (x1, y1) = persons_center[index]
        (x2, y2) = IDUtilities.get_center(person)
        a = np.array([x1, y1])
        b = np.array([x2, y2])
        u = b - a
        return np.linalg.norm(u)

    @staticmethod
    def is_overlap(persons, index):
        [x1, y1, x2, y2] = persons[index]
        for i, person in enumerate(persons):
            if (index == i):
                continue
            if (max(person[0], x1) <= min(person[2], x2) and max(person[1], y1) <= min(person[3], y2)):
                return True
        return False

    @staticmethod
    def cos_similarity(X, Y):

        # Cosine similarity
        # reference document: https://github.com/kodamap/person_reidentification
        m = X.shape[0]
        Y = Y.T
        return np.dot(X, Y) / (np.linalg.norm(X.T, axis=0).reshape(m, 1) * np.linalg.norm(Y, axis=0))

    @staticmethod
    def get_similarity(identifies_ref, identifies_new):

        id_list = []
        identifies_db = []

        for k in identifies_ref.keys():

            id_list.append(k)

            # Take mean values of similarity
            identify_mean = np.mean(identifies_ref[k], axis=0).tolist()
            identifies_db.append(identify_mean)

        identifies_db = np.array(identifies_db)
        similaritys_all = IDUtilities.cos_similarity(identifies_new, identifies_db)
        similaritys_all[np.isnan(similaritys_all)] = 0
        ids = np.nanargmax(similaritys_all, axis=1)

        similaritys = []

        # loop each row
        for i in range(similaritys_all.shape[0]):
            similaritys.append(similaritys_all[i, ids[i]])

        return similaritys, ids.tolist()

    @staticmethod
    def get_similarity_nn(identifies_ref, identifies_new, n: int = 2):

        id_list = []
        identifies_db = []

        for k in identifies_ref.keys():
            for iden in identifies_ref[k]:
                id_list.append(k)
                identifies_db.append(iden)

        identifies_db = np.array(identifies_db)

        # Maybe there is not enough data
        if identifies_db.shape[0] > n:
            n = n
        else:
            n = identifies_db.shape[0]

        similaritys = cosine_similarity(identifies_new, identifies_db, dense_output=True)
        similaritys[np.isnan(similaritys)] = 0

        index = np.argsort(-similaritys, axis=1)[:, :n]

        ids = []
        similaritys_mean = []

        # loop each row
        for i in range(index.shape[0]):

            ind_row = index[i]

            # Take majority
            c = Counter(np.array(id_list)[ind_row])
            ids.append(int(c.most_common(1)[0][0]))

            # Take all similarity value found and mean them
            temp = np.mean(similaritys[i, ind_row])
            similaritys_mean.append(temp)

        return similaritys_mean, ids

class DatabaseIDs:

    def __init__(self, path_database_id: str):

        self.path_database_id = path_database_id

        self.load()

        self.init_centers()

    def init_centers(self):

        self.centers = {}

        if len(self.identifies.keys()) == 0:
            pass
        else:
            for k in self.identifies.keys():
                self.centers[k] = [0, 0]  # Reset center for every new video

    def load(self):

        self.ids = {}

        self.images = {}
        self.names = {}
        self.identifies = {}

        self.numbers = []

        # Load all ids json
        for file in os.listdir(self.path_database_id):
            if file.endswith('.json'):
                # Deserialize json
                with open(self.path_database_id + file, 'r') as openfile:
                    # Reading from json file
                    s = openfile.read()

                id = ID(s=s)

                self.names[id.number] = id.name
                self.identifies[id.number] = id.identify
                self.numbers.append(id.number)

                print("ID number in database: " + str(id.number) + " - Name + " + str(id.name))

                self.ids[id.number] = id

                # Get corresponding image
                img = cv2.imread(self.path_database_id + str(id.number) + ".jpg")
                self.images[id.number] = img

    def get_ids(self, identifies_new, persons, image_boxes,
                DETECTION_THRESHOLD_UP: float = 0.95, DETECTION_THRESHOLD_DOWN: float = 0.4,
                DISTANCE_THRESHOLD: int = 100, DETECTION_N: int = 5):

        if identifies_new.size == 0:
            return []

        # If empty it mean that the database does not exist yet. Add all person detected in the current image
        if len(self.identifies.keys()) == 0:
            for i in range(identifies_new.shape[0]):

                identify = identifies_new[i].tolist()
                person_number = self.add_new(identify=identify, image=image_boxes[i])
                self.update_center(person_box=persons[i], person_number=person_number)

            # Reload the database
            self.load()

        similaritys, ids = IDUtilities.get_similarity(identifies_ref=self.identifies, identifies_new=identifies_new)
        similaritys, ids = IDUtilities.get_similarity_nn(identifies_ref=self.identifies, identifies_new=identifies_new, n=DETECTION_N)

        print("")

        for i, similarity in enumerate(similaritys):

            persionId = ids[i]
            d = IDUtilities.get_distance(person=persons[i], persons_center=self.centers, index=persionId)

            print("Person similarity: " + str(similarity) + " - distance: " + str(d))

            # If similarity is greater than DETECTION_THRESHOLD and there is no overlap, the identification information is updated
            if similarity > DETECTION_THRESHOLD_UP:
                if IDUtilities.is_overlap(persons, i) == False:
                    print("Person updated in the database with id: " + str(persionId))
                    identify = identifies_new[i].tolist()
                    self.update_existing(person_number=persionId, identify=identify)
                    self.update_center(person_number=persionId, person_box=persons[i])

            # If similarity is less than 0.5 and the distance is far, register a new ID
            elif similarity < DETECTION_THRESHOLD_DOWN:
                if d > DISTANCE_THRESHOLD:

                    identify = identifies_new[i].tolist()
                    person_number = self.add_new(identify=identify, image=image_boxes[i])
                    print("Person added to the database with id: " + str(person_number))
                    self.update_center(person_box=persons[i], person_number=person_number)
                    # Reload the full database
                    self.load()

        return ids

    def update_center(self, person_number: int, person_box):

        # Update center position
        center = IDUtilities.get_center(person_box=person_box)
        self.centers[person_number] = [center[0], center[1]]

    def update_existing(self, person_number: int, identify: list):

        # Add identify information to the list of list of similarity
        self.ids[person_number].identify.append(identify)
        self.ids[person_number].save(path_folder=self.path_database_id)

    def add_new(self, identify: list, image):

        # Create new name if not provided
        if self.numbers:
            person_number_new = max(self.numbers) + 1
        else:
            person_number_new = 0

        name_new = "Person " + str(person_number_new)

        id = ID()
        id.number = person_number_new
        id.name = name_new

        # TODO: temp adding twice the identify to start working artificially on list of list
        id.identify = [identify, identify]
        id.save(path_folder=self.path_database_id)

        self.numbers.append(person_number_new)

        print("New ID number in database: " + str(id.number) + " - Name + " + str(id.name))

        # Save image box as jpg
        cv2.imwrite(self.path_database_id + str(id.number) + ".jpg", image)

        return person_number_new

    def delete_all(self):

        shutil.rmtree(self.path_database_id)
