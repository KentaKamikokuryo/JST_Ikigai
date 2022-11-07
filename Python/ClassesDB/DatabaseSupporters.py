from ClassesDB.DatabaseIDs import *
import os
import cv2
import json
import shutil

# TODO: It could be better to integrate existing database class

class IDPersonalized:

    identify_face: list
    identify_body: list
    number: int
    name: str

    def __init__(self, s=None):

        if s in None:
            pass
        else:
            self.__dict__ = json.loads(s)

    def _to_json(self):

        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def save(self, path_folder: str):

        s = self._to_json()

        with open(path_folder + str(self.number) + ".json", "w") as outfile:
            outfile.write(s)

class Database:

    def __init__(self, path_database_supp: str, name: str):

        self.path_database = path_database_supp  # TODO: set the new ReID folder
        self.identifies_face = []
        self.identifies_body = []
        self.image_boxes_face = []
        self.image_boxes_body = []
        self.name = name

        self._load()

    def _load(self):

        self.ids = {}
        self.images = {}
        self.names = {}
        self.identifies = {}

        self.numbers = []

        for file in os.listdir(self.path_database):
            if file.endswith(".json"):
                with open(self.path_database + file, "r") as openfile:
                    s = openfile.read()

            _id = IDPersonalized(s=s)

            self.names[_id.number] = _id.name
            self.ids[_id.number] = _id
            self.identifies_face[_id.number] = _id.identify_face
            self.identifies_body[_id.number] = _id.identify_body
            self.numbers.append(_id.number)

            print("ID number in Database: {} - Name: {}".format(_id.number, _id.name))

            img = cv2.imread(self.path_database + str(_id.number) + ".jpg")
            self.images[_id.number] = img

    def add_id_face(self, identify_face, image_box_face):

        self.identifies_face.append(identify_face)
        self.image_boxes_body.append(image_box_face)

    def add_id_body(self, identify_body, image_box_body):

        self.identifies_body.append(identify_body)
        self.image_boxes_body.append(image_box_body)

    def save(self):

        if self.name not in self.names.values():

            if self.numbers:
                person_number_new = max(self.numbers) + 1
            else:
                person_number_new = 0

            _id = IDPersonalized()
            _id.number = person_number_new
            _id.name = self.name
            _id.identify_face = self.identifies_face
            _id.identify_body = self.identifies_body

            _id.save(path_folder=self.path_database)

            self.numbers.append(person_number_new)

            print("New ID number in database: {} - Name: {}".format(_id.number, _id.name))

            cv2.imwrite(self.path_database + str(_id.number) + "face.jpg", self.image_boxes_face[0])
            # TODO:
            cv2.imwrite(self.path_database + str(_id.number) + "body.jpg", self.image_boxes_body[0])

        else:

            # TODO: Keep thinking about here

            pass

    def delete(self):

        shutil.rmtree(self.path_database)





