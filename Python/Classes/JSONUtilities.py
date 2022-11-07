import json
import rapidjson
import numpy as np

class JSONUtilities:

    @staticmethod
    def save_dictionary_as_json(data_dict: dict, folder_path: str, filename: str):

        path = folder_path + filename + ".json"

        with open(path, 'w') as f:
            for i in data_dict.keys():
                data = data_dict[i]
                data["Frame"] = int(i)
                json_object = json.dumps(data)
                f.write(json_object)
                f.write("\n")

    @staticmethod
    def load_json_as_dictionary(folder_path: str, filename: str):

        path = folder_path + filename + ".json"

        f = open(path, mode='r', encoding='utf8', newline='')

        data = []

        for i, line in enumerate(f):
            data.append(rapidjson.loads(line))
        f.close()

        return data

    @staticmethod
    def list_dict_to_dict_list(dict_list: list):

        # Will change list of dictionary to dictionary of numpy array
        keys = list(dict_list[0].keys())
        data_dict = {}

        for k in keys:
            data_dict[k] = []

        # Loop through list of dictionary
        for d in dict_list:
            for k in keys:
                data_dict[k].append(d[k])

        return data_dict