import copy
import numpy as np
from Classes.Drawing import DrawingUtilities, Drawing
from Classes.Video import VideoUtilities
from Kenta.TempClassesML.Factories import DetectorFactory


class VideoManager:

    def __init__(self, plot_parameters):

        self.SCALE_IMAGE = plot_parameters["scale_image"]
        self.SCALE_TEXT = plot_parameters["scale_text"]
        self.MARGIN = plot_parameters["margin"]
        self.THICKNESS = plot_parameters["thickness"]

        self._image_height = 0
        self._image_width = 0
        self._image_list = []

        self.colors = DrawingUtilities.set_colors_tracking(tracking_max=50)

    def save_video(self, video_save_folder: str, video_name: str, model_names: str):

        save_path = video_save_folder + video_name + "_{}.mp4".format(model_names)
        VideoUtilities.save_images_to_video(images=self._image_list,
                                            save_path=save_path,
                                            image_height=self._image_height,
                                            image_width=self._image_width)

    def write_bbox(self,
                   data_frame_dict: dict,
                   person_results: dict,
                   conf: dict,
                   fac: DetectorFactory,
                   is_info: bool = False):

        model_name = fac.model_name
        model_color = fac.color
        pos = fac.pos

        # image_list = []
        self._image_list = []
        processed_data_frame_dict = {}
        n_frame = len(data_frame_dict.keys())

        for i in range(n_frame):

            image = copy.deepcopy(data_frame_dict[i])
            self._image_height, self._image_width, _ = image.shape

            if is_info:

                image = DrawingUtilities.put_highlight_text(image=image,
                                                            text="Bounding Box: {}".format(model_name),
                                                            org=(0, 30 * pos),
                                                            scale_text=self.SCALE_TEXT + 1,
                                                            color=model_color,
                                                            thickness=self.THICKNESS)

            image = self._draw_bbox(image=image,
                                    persons=person_results[i],
                                    conf=conf[i],
                                    color=model_color)

            self._image_list.append(image)
            processed_data_frame_dict[i] = image
            print("{} - Drawing BBox at frame: {}/{}".format(model_name, i, n_frame))

        return processed_data_frame_dict

    def write_bbox_ids(self,
                       data_frame_dict,
                       person_results,
                       ids_results,
                       conf):

        self._image_list = []
        processed_data_frame_dict = {}
        n_frame = len(data_frame_dict.keys())

        for i in range(n_frame):

            image = copy.deepcopy(data_frame_dict[i])
            self._image_height, self._image_width, _ = image.shape

            image = self._draw_bbox_ids(image=image,
                                        persons=person_results[i],
                                        conf=conf[i],
                                        ids_results=ids_results[i])

            self._image_list.append(image)
            processed_data_frame_dict[i] = image
            print("Drawing BBox and IDs at frame: {}/{}".format(i, n_frame))

        return processed_data_frame_dict

    def write_mediapipe_landmarks(self,
                                  data_frame_dict,
                                  person_results,
                                  ids_results,
                                  mpl_results,
                                  mp,
                                  file_name: str):

        self._image_list = []
        processed_data_frame_dict = {}
        n_frame = len(data_frame_dict.keys())

        for i in range(n_frame):

            image = copy.deepcopy(data_frame_dict[i])
            self._image_height, self._image_width, _ = image.shape

            if file_name == "Face_IDs":

                image = self._draw_facemesh(image=image,
                                            persons=person_results[i],
                                            ids_results=ids_results[i],
                                            mpl_results=mpl_results[i],
                                            mp=mp)

            elif file_name == "IDs":

                image = self._draw_pose_landmark(image=image,
                                                 persons=person_results[i],
                                                 ids_results=ids_results[i],
                                                 mpl_results=mpl_results[i],
                                                 mp=mp)

            else:

                import sys
                sys.exit(1)

            self._image_list.append(image)
            processed_data_frame_dict[i] = image
            print("Drawing face mesh at frame: {}/{}".format(i, n_frame))

        return processed_data_frame_dict

    def _draw_bbox(self, image, persons, conf, color):

        for i, person in enumerate(persons):

            start_point = (person[0] - self.MARGIN, person[1] - self.MARGIN)
            end_point = (person[2] + self.MARGIN, person[3] + self.MARGIN)

            image = DrawingUtilities.put_highlight_rectangle(image=image,
                                                             pt1=start_point,
                                                             pt2=end_point,
                                                             color=color,
                                                             thickness=self.THICKNESS)

            image = DrawingUtilities.put_highlight_text(image=image,
                                                        text="conf: {:.2f}".format(conf[i]),
                                                        org=(person[0] - self.MARGIN, person[3] + self.MARGIN * 3),
                                                        scale_text=self.SCALE_TEXT,
                                                        color=color,
                                                        thickness=self.THICKNESS)

        return image

    def _draw_bbox_ids(self, image, persons, conf, ids_results):

        for i, person in enumerate(persons):

            color = self.colors[int(ids_results[i])]

            start_point = (person[0] - self.MARGIN, person[1] - self.MARGIN)
            end_point = (person[2] + self.MARGIN, person[3] + self.MARGIN)

            image = DrawingUtilities.put_highlight_rectangle(image=image,
                                                             pt1=start_point,
                                                             pt2=end_point,
                                                             color=color,
                                                             thickness=self.THICKNESS)

            image = DrawingUtilities.put_highlight_text(image=image,
                                                        text="{}: {:.2f}".format(ids_results[i], conf[i]),
                                                        org=(person[0] - self.MARGIN, person[1] - self.MARGIN),
                                                        scale_text=self.SCALE_TEXT,
                                                        color=color,
                                                        thickness=self.THICKNESS)

        return image

    def _draw_facemesh(self, image, persons, ids_results, mpl_results, mp):

        for i, face in enumerate(persons):

            color = self.colors[int(ids_results[i])]
            color_inv = (255 - color[0], 255 - color[1], 255 - color[2])

            id = ids_results[i]

            img = image[(int(face[1]) - self.MARGIN):(int(face[3]) + self.MARGIN),
                  (int(face[0]) - self.MARGIN):(int(face[2]) + self.MARGIN)]

            if mpl_results[id]:

                for face_landmarks in mpl_results[id]:

                    Drawing.mp_drawing.draw_landmarks(image=img,
                                                      landmark_list=face_landmarks,
                                                      connections=mp.FACEMESH_CONTOURS,
                                                      landmark_drawing_spec=Drawing.mp_drawing.DrawingSpec(color=color,
                                                                                                           thickness=1,
                                                                                                           circle_radius=3),
                                                      connection_drawing_spec=Drawing.mp_drawing.DrawingSpec(
                                                      color=Drawing.color_white, thickness=1))

                    Drawing.mp_drawing.draw_landmarks(image=img,
                                                      landmark_list=face_landmarks,
                                                      connections=mp.FACEMESH_IRISES,
                                                      landmark_drawing_spec=None,
                                                      connection_drawing_spec=Drawing.mp_drawing.DrawingSpec(
                                                          color=color_inv, thickness=3))

        return image

    def _draw_pose_landmark(self, image, persons, ids_results, mpl_results, mp):

        for i, face in enumerate(persons):

            color = self.colors[int(ids_results[i])]
            color_inv = (255 - color[0], 255 - color[1], 255 - color[2])

            id = ids_results[i]

            img = image[(int(face[1]) - self.MARGIN):(int(face[3]) + self.MARGIN),
                  (int(face[0]) - self.MARGIN):(int(face[2]) + self.MARGIN)]

            if mpl_results[id]:

                Drawing.mp_drawing.draw_landmarks(image=img,
                                                  landmark_list=mpl_results[id],
                                                  connections=mp.POSE_CONNECTIONS,
                                                  landmark_drawing_spec=Drawing.mp_drawing.DrawingSpec(color=color,
                                                                                                       thickness=3,
                                                                                                       circle_radius=5),
                                                  connection_drawing_spec=Drawing.mp_drawing.DrawingSpec(
                                                      color=Drawing.color_white, thickness=3))


        return image


