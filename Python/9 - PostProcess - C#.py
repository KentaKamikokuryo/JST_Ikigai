import copy
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from Classes.PathInfo import PathInfo
from Classes.Video import *
import random
import numpy as np
from Classes.JSONUtilities import JSONUtilities
import itertools
from collections import Counter

class Analysis:

    def __init__(self, video_name):

        self.video_name = video_name

        self.path_info = PathInfo()
        self.path_info.set_results_folder(folder_path=self.path_info.path_data_test, name=Path(video_name).stem)

        # Load IDs
        self.results_frame = JSONUtilities.load_json_as_dictionary(folder_path=self.path_info.results_folder, filename="Face_Emotions")
        self.results_ids = JSONUtilities.list_dict_to_dict_list(dict_list=self.results_frame)
        # keys: ['IDs', 'Box', 'Emotions', 'Emotions_prob', 'Frame']

        self.possible_emotions = ["happy", "sad", "surprise", "anger", "neutral"]

    def compute_percentage_emotions(self):

        # Get all ID detected first
        temp = np.concatenate([l for l in self.results_ids["IDs"]])
        self.unique_ids = np.unique(temp)

        data_emotion_id = {}

        # Get percentage of emotion at each frame
        for id in self.unique_ids:

            data_emotion_id[id] = {}

            emotions = []
            emotions_prob = []
            frames = []

            for frame in range(len(self.results_ids["Frame"])):

                # Find id index to get corresponding data
                ids_at_frame = self.results_ids["IDs"][frame]
                index = [index for index in range(len(ids_at_frame)) if ids_at_frame[index] == id]

                if len(index) == 1:

                    emotions.append(self.results_ids["Emotions"][frame][index[0]])
                    emotions_prob.append(self.results_ids["Emotions_prob"][frame][index[0]])
                    frames.append(frame)

                else:

                    emotions.append(None)
                    emotions_prob.append(0.0)

            data_emotion_id[id]["Emotions"] = emotions
            data_emotion_id[id]["Emotions_prob"] = emotions_prob
            data_emotion_id[id]["Frame"] = frames

        # Get number of frame for each id and each emotion
        # Get percentage of emotion for each id
        data_emotion_n_frame_id = {}
        data_emotion_percentage_id = {}

        for id in self.unique_ids:

            counter = Counter(data_emotion_id[id]["Emotions"])
            emotion_n_frame = {}
            emotion_percentage = {}

            for emotion in self.possible_emotions:

                if emotion in counter.keys():
                    emotion_n_frame[emotion] = counter[emotion]
                else:
                    emotion_n_frame[emotion] = 0

            data_emotion_n_frame_id[id] = emotion_n_frame

            s = sum(emotion_n_frame.values())
            for emotion, count in emotion_n_frame.items():
                pct = count * 100.0 / s
                emotion_percentage[emotion] = pct

            data_emotion_n_frame_id[id] = emotion_n_frame
            data_emotion_percentage_id[id] = emotion_percentage

        # Get cumulative values at each frame for each emotion and each id
        data_emotion_percentage_at_frame_id = {}

        for id in self.unique_ids:

            emotion_percentage_at_frame = []

            for frame in range(len(self.results_ids["Frame"])):

                counter = Counter(data_emotion_id[id]["Emotions"][:frame])

                emotion_n_frame = {}
                emotion_percentage = {}

                for emotion in self.possible_emotions:

                    if emotion in counter.keys():
                        emotion_n_frame[emotion] = counter[emotion]
                    else:
                        emotion_n_frame[emotion] = 0

                s = sum(emotion_n_frame.values())

                if s > 0:  # no emotion yet
                    for emotion, count in emotion_n_frame.items():
                        pct = count * 100.0 / s
                        emotion_percentage[emotion] = pct
                else:
                    for emotion, count in emotion_n_frame.items():
                        emotion_percentage[emotion] = 0.0

                emotion_percentage_at_frame.append(emotion_percentage)

            data_emotion_percentage_at_frame_id[id] = emotion_percentage_at_frame

        # Data to keep
        self.data_emotion_percentage_id = data_emotion_percentage_id
        self.data_emotion_percentage_at_frame_id = data_emotion_percentage_at_frame_id

    def combine(self):

        self.results = {}

        for frame in range(len(self.results_ids["Frame"])):

            self.results[frame] = {}

            emotion_of_id_at_frame = []
            for id in self.data_emotion_percentage_at_frame_id.keys():
                emotion_of_id_at_frame.append(self.data_emotion_percentage_at_frame_id[id][frame])

            self.results[frame]["IDs"] = self.unique_ids.tolist()
            self.results[frame]["Emotion_p"] = emotion_of_id_at_frame

    def analyse(self):

        # Compute percentage that the person is detected in total
        ids = list(itertools.chain.from_iterable(self.results_ids["IDs"]))
        n_frame = len(self.results_ids["Frame"])
        unique, counts = np.unique(ids, return_counts=True)

        self.results_ids_p = {}
        for id, count in zip(unique, counts):
            self.results_ids_p[str(id)] = count / n_frame

    def plot_ids_percentage(self):

        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(111)

        # creating the bar plot
        c = ['blue', 'green', 'red']
        x = self.results_ids_p.keys()
        y = self.results_ids_p.values()
        ax1.bar(x, y, width=0.4, color=c)
        ax1.set_xlabel("IDs")
        ax1.set_ylabel("Percentage (%)")
        ax1.set_title("Percentage that person is detected in: " + self.video_name)

    def save(self):

        # Combine dictionary
        JSONUtilities.save_dictionary_as_json(data_dict=self.results,
                                                            folder_path=self.path_info.results_folder,
                                                            filename="Face_Emotions_p")

video_name = "Video_test_0.mp4"

analysis = Analysis(video_name=video_name)
analysis.compute_percentage_emotions()
analysis.combine()
analysis.analyse()
analysis.plot_ids_percentage()
analysis.save()
