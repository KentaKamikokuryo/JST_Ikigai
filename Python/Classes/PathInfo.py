import os
from pathlib import Path

def is_debug():
    import sys

    gettrace = getattr(sys, 'gettrace', None)

    if gettrace is None:
        return False
    else:
        v = gettrace()
        if v is None:
            return False
        else:
            return True

class PathInfo:

    def __init__(self, path_parent_project=None):

        if path_parent_project == None:

            print(is_debug())

            if is_debug():
                self.path_parent_project = os.getcwd()
            else:
                self.path_parent_project = os.getcwd() + "\\Python"
        else:
            self.path_parent_project = path_parent_project

        self.path_data_raw = self.path_parent_project + "\\Data_raw\\"
        self.path_data_sync = self.path_parent_project + "\\Data_process\\"

        print("path_parent_project: " + self.path_parent_project)
        print("path_data_raw: " + self.path_data_raw)
        print("path_data_sync: " + self.path_data_sync)

        if not os.path.exists(self.path_data_sync):
            os.makedirs(self.path_data_sync)

        # Models path
        self.path_model = self.path_parent_project + "\\Models\\"

        if not os.path.exists(self.path_model):
            os.mkdir(self.path_model)

        print("path_model: " + self.path_model)

        self.sub_folder_name = "TestVideo"

        self.path_data_test = self.path_parent_project + "\\" + self.sub_folder_name + "\\"

        print("path_data_test: " + self.path_data_test)

        self.search_folder = self.path_parent_project + "\\" + "Search_ReID" + "\\"
        self.final_folder = self.path_parent_project + "\\" + "Final_ReID" + "\\"

        print("search_folder: " + self.search_folder)
        print("final_folder: " + self.final_folder)

        self.path_database_video_face = self.path_parent_project + "\\" + "DatabaseVideo" + "\\" + "Face" + "\\"

        print("path_database_video: {}".format(self.path_database_video_face))

    def set_results_folder(self, name: str = ""):

        if not os.path.exists(self.final_folder):
            os.mkdir(self.final_folder)

        self.path_database_id = self.final_folder + "\\DatabaseIDs\\"
        self.path_database_face_id = self.final_folder + "\\DatabaseFaceIDs\\"

        self.results_folder = self.final_folder + name + "\\"

        if not os.path.exists(self.path_database_id):
            os.mkdir(self.path_database_id)

        if not os.path.exists(self.path_database_face_id):
            os.mkdir(self.path_database_face_id)

        if not os.path.exists(self.results_folder):
            os.mkdir(self.results_folder)

    def set_results_folder_hyper(self, count: int = 0, name: str = ""):

        if not os.path.exists(self.search_folder):
            os.mkdir(self.search_folder)

        self.search_count_folder = self.search_folder + "Search_" + str(count) + "\\"

        self.path_database_id_search = self.search_count_folder + "\\DatabaseIDs\\"
        self.path_database_face_id_search = self.search_count_folder + "\\DatabaseFaceIDs\\"

        self.results_folder = self.search_count_folder + name + "\\"

        if not os.path.exists(self.search_count_folder):
            os.mkdir(self.search_count_folder)

        if not os.path.exists(self.path_database_id_search):
            os.mkdir(self.path_database_id_search)

        if not os.path.exists(self.path_database_face_id_search):
            os.mkdir(self.path_database_face_id_search)

        if not os.path.exists(self.results_folder):
            os.mkdir(self.results_folder)

    def delete_database_id(self):

        pass

