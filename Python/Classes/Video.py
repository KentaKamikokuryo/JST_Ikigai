import copy
import os
import cv2
from pathlib import Path

class VideoUtilities:

    def __init__(self):

        pass

    @staticmethod
    def save_images_to_video(images: list, save_path: str, image_width: int = 1280, image_height: int = 720):

        if len(images) > 0:

            size = (image_width, image_height)
            fps = 25
            out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'XVID'), fps, size)

            for i in range(len(images)):
                out.write(images[i])

            out.release()

            print("Video with landmark with: " + save_path)

        else:

            print("Video with landmark cannot be saved - No Landmark")

    @staticmethod
    def save_video_from_images_folder(path_images_folder: str, save_path: str, fps: int = 25):

        # Make video
        image_names = os.listdir(path_images_folder)
        frame = cv2.imread(os.path.join(path_images_folder, image_names[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(save_path, 0, fps, (width, height))

        n_frame = len(image_names)

        for frame in range(n_frame):
            print("Processing image " + image_names[frame] + " to video - Remaining frame: " + str(n_frame - frame))
            video.write(cv2.imread(os.path.join(path_images_folder, image_names[frame])))

        print("Have done saving the video!!")

        cv2.destroyAllWindows()
        video.release()

    @staticmethod
    def save_video_box():

        pass

    @staticmethod
    def save_video_box_with_ids():

        pass

class ProcessVideo:

    def __init__(self, video_path):

        self.video_path = video_path
        self.video_name = Path(self.video_path).stem

        if not os.path.exists(self.video_path):
            print("Video not found at " + self.video_path)

    def read(self, n_frame:  int = 100, resize_param: int = None):

        cap = cv2.VideoCapture(self.video_path)

        self.n_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if not n_frame > self.n_frame:
            self.n_frame = n_frame

        self.data_frame_dict = {}

        # Extract all frames
        for i in range(self.n_frame):
            print("Extracting frame " + str(i) + "/" + str(self.n_frame))
            cap.set(1, i)
            ret, frame = cap.read()

            image_height, image_width = frame.shape[:2]
            if resize_param:
                size = (image_width * resize_param, image_height * resize_param)
                frame = cv2.resize(frame, size)

            frame.flags.writeable = True
            self.data_frame_dict[i] = frame

            self.image_height, self.image_width, _ = frame.shape

        return copy.deepcopy(self.data_frame_dict)


class ProcessVideoV2:
    """
    This class can load a video that is extracted by specifying the start frame number, end frame number, frame length,
    start seconds, end seconds, or time length.

    In the case of the value is input for each parameter, the value will be used in the following order of priority:

    start_frame(start frame number) = n_frame(frame length) < end_frame(end frame number) < start_sec(start seconds) =
    seconds(time length) < end_sec(end seconds)
    (* the above parameters are input in the read() method.)
    """

    def __init__(self, video_path):

        self.video_path = video_path
        self.video_name = Path(self.video_path).stem

        assert os.path.exists(self.video_path), "Video not found at {0}!!".format(self.video_path)

    def read(self,
             start_frame: int = 0, end_frame: int = None, n_frame:  int = 100,
             start_sec: float = None, end_sec: float = None, seconds: float = None,
             resize_param: int = None):
        """
        Load the specified video with the specified frame or time, and return the extracted video.

        :param start_frame: start frame number, is ignored when start_sec is input.
        :param end_frame: end frame number, is ignored when end_sec is input.
        :param n_frame: frame length, is ignored when end_frame is input
        :param start_sec: start seconds
        :param end_sec: end seconds
        :param seconds: time length, is ignored when end_sec is input.
        :param resize_param:
        :return: extracted video(dict)
        """

        self.cap = cv2.VideoCapture(self.video_path)

        self._cap_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._cap_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self._cap_seconds = float(self.cap_frames) / self._cap_fps
        self._cap_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._cap_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        print(f"Have loaded the video at {self.video_path}")
        print(f"video format: {self.cap_frames} frames, {self.cap_seconds} sec, {self.cap_fps} fps")

        self.start_frame = int(self.cap_fps * start_sec) if start_sec is not None else start_frame
        if self.start_frame > self.cap_frames:
            self.start_frame = 0

        if end_sec is not None:
            self.n_frame = int(self.cap_fps * end_sec) - self.start_frame
        elif seconds is not None:
            self.n_frame = int(self.cap_fps * seconds)
        elif end_frame is not None:
            self.n_frame = end_frame - self.start_frame
        else:
            self.n_frame = n_frame
        if (self.start_frame+self.n_frame > self.cap_frames) or (self.start_frame+self.n_frame < self.start_frame):
            self.n_frame = self.cap_frames - self.start_frame

        print(f"Start extracting from {self.start_frame} to {self.start_frame+self.n_frame}...")

        self.video_frame_dict = {}

        # Extract all frames
        for i, f in enumerate(range(self.start_frame, self.start_frame+self.n_frame)):

            print(f"Extracting frame {i}/{self.n_frame} ({self.start_frame}-{self.start_frame+self.n_frame})")
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ret, frame = self.cap.read()

            image_height, image_width = frame.shape[:2]
            if resize_param:
                size = (image_width * resize_param, image_height * resize_param)
                frame = cv2.resize(frame, size)

            frame.flags.writeable = True
            self.video_frame_dict[i] = frame

        return copy.deepcopy(self.video_frame_dict)

    # region properties
    @property
    def cap_frames(self):
        return self._cap_frames

    @property
    def cap_fps(self):
        return self._cap_fps

    @property
    def cap_seconds(self):
        return self._cap_seconds

    @property
    def cap_height(self):
        return self._cap_height

    @property
    def cap_width(self):
        return self._cap_width
    # endregion
