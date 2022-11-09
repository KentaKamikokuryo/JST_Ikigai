from ClassesDB.DatabaseIDs import *
import os
import cv2
import json
import shutil

# TODO: It could be better to integrate existing database class

class Database:

    def __init__(self, path_database_supp: str, name: str, id: int):

        self.path_database = path_database_supp  # TODO: set the new ReID folder
        self.name = name
        self.path_database_json = self.path_database + str(id) + ".json"

        self.identifies = []
        self.image_boxes = []
        self.number = id

    def load(self):
        """
        This function is used before calling get_id function
        :return:
        """

        self.identifies = []
        self.image_boxes = []

        if os.path.exists(self.path_database_json):
            with open(self.path_database_json, 'r') as openfile:
                s = openfile.read()

            _id = ID(s=s)
            self.identifies = _id.identify  # Get all identifies

            print("ID number in Database: {} - Name: {}".format(_id.number, _id.name))

            for i in range(len(self.identifies)):
                img = cv2.imread(self.path_database + str(self.number) + "_" + str(i) + ".jpg")
                self.image_boxes.append(img)

    def get_id(self,
               identify,
               image_box,
               DETECTION_THRESHOLD: float = 0.9,
               DETECTION_N: int = 5):

        if identify.size == 0:
            return []
        if len(image_box) == 0:
            return []

        if len(self.identifies) == 0:

            self.identifies.append(identify[0].tolist())
            self.image_boxes.append(image_box[0])

        similaritys, _ = IDUtilities.get_similarity_nn(identifies_ref={i: [self.identifies[i]] for i in range(len(self.identifies))},
                                                       identifies_new=identify,
                                                       n=DETECTION_N)

        similarity = similaritys[0]

        print("")
        print("Name: {} - Similarity: {}".format(self.name, similarity))

        if similarity < DETECTION_THRESHOLD:

            self.identifies.append(identify[0].tolist())
            self.image_boxes.append(image_box[0])

    def get_id_simple(self,
                      identify,
                      image_box):

        if identify.size == 0 or len(image_box) == 0:
            return []

        self.identifies.append(identify[0].tolist())
        self.image_boxes.append(image_box[0])

    def save(self, image_frame_per_person: int = 0):

        _id = ID()
        _id.name = self.name
        _id.identify = self.identifies
        _id.number = self.number
        _id.save(path_folder=self.path_database)

        print("New ID number in database: {} - Name: {}".format(_id.number, _id.name))
        #
        # for i, image_box in enumerate(self.image_boxes):
        #
        #     cv2.imwrite(self.path_database + str(_id.number) + "_" + str(i) + ".jpg", image_box)
        cv2.imwrite(self.path_database + str(self.number) + ".jpg", self.image_boxes[image_frame_per_person])

    def delete(self):

        shutil.rmtree(self.path_database)





