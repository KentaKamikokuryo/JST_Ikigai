import matplotlib.pyplot as plt
import numpy as np
import cv2
import copy
import os
import mediapipe as mp
import imageio
import shutil
from Classes.PathInfo import PathInfo
from Classes.Video import ProcessVideo
from Classes.Pose import BiomechanicsInformation, PresenceVisibility, PoseFeatures, BiomechanicsImageInformation
from Classes.Drawing import Drawing
from Classes.Drawing import DrawingUtilities
from Classes.Video import VideoUtilities
from Kenta.TempClasses.PathInfo import KentaPathInfo

"""
For test, the code gets bunch of frames, and then compute biomecanics information
"""
import seaborn as sns
sns.set(style='ticks', rc={"grid.linewidth": 0.1})
sns.set_context("paper", font_scale=2.5)
color = sns.color_palette("Set2", 6)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.ioff()

mp_drawing = mp.solutions.drawing_utils  # Drawing helpers
mp_holistic = mp.solutions.holistic  # Mediapipe Solutions

holistic_detection_parameters = dict(static_image_mode=False,
                                     min_detection_confidence=0.5,
                                     model_complexity=1,
                                     min_tracking_confidence=0.4,
                                     smooth_landmarks=True)

def _quiver(ax, origin, vector, color):
    ax.quiver(origin[0], origin[1], origin[2], vector[0], vector[1], vector[2], color=color, length=0.2)

path_info = PathInfo()
video_name = "Experimental_Video"
video_fmt = ".mp4"
video_path = path_info.path_data_test + video_name + video_fmt
# video_path = path_info.path_data_test + "Video_test_2.mp4"

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
processVideo = ProcessVideo(video_path=video_path)
data_frame_dict = processVideo.read(n_frame=500)

n_frame = len(data_frame_dict.keys())


# results dictionary
results = {}
results_w = {}
results_w_i = {}
results_v = {}
results_i = {}

results_biomeca = {}

image_list = []

# loop in the frames got
for i in range(n_frame):

    image = copy.deepcopy(data_frame_dict[i])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_height, image_width, _ = image.shape

    with mp_holistic.Holistic(**holistic_detection_parameters) as holistic:

        print("Computing 3D pose biomechanics information at frame: " + str(i) + "/" + str(n_frame))

        # Store all results to dictionary
        results[i] = holistic.process(image)
        n_landmark = len(results[i].pose_world_landmarks.landmark)
        landmarks_w = results[i].pose_world_landmarks.landmark
        landmarks = results[i].pose_landmarks.landmark

        results_i[i] = np.array([[p.x * image_width, p.y * image_height] for p in results[i].pose_landmarks.landmark])
        results_w[i] = np.array([[-landmarks_w[k].z, landmarks_w[k].x, -landmarks_w[k].y] for k in range(n_landmark)])
        results_w_i[i] = np.array([[landmarks[k].x, landmarks[k].y, landmarks[k].z] for k in range(n_landmark)])
        results_v[i] = np.array([[landmarks_w[k].visibility] for k in range(n_landmark)])

        presenceVisibility = PresenceVisibility(results_v=results_v[i], threshold=0.8)
        biomechanicsInformation = BiomechanicsInformation(results_hollistic_w=results_w[i],
                                                          presenceVisibility=presenceVisibility,
                                                          display=False)
        biomechanicsImageInformation = BiomechanicsImageInformation(results_i=results_i[i])
        results_biomeca[i] = biomechanicsInformation.dict

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mp_drawing.draw_landmarks(
            image,
            results[i].pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=Drawing.color_green, thickness=2, circle_radius=4),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=Drawing.color_white, thickness=2)
        )

        for key in biomechanicsImageInformation.dict.keys():

            if "r" in key[-1]:
                x = biomechanicsImageInformation.dict[key][0] - 60
            else:
                x = biomechanicsImageInformation.dict[key][0] + 10

            cv2.putText(img=image,
                        text="{:.1f}".format(results_biomeca[i][key]),
                        org=(int(x), int(biomechanicsImageInformation.dict[key][1])),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.6,
                        color=Drawing.color_black,
                        thickness=2,
                        lineType=cv2.LINE_AA)

            cv2.putText(img=image,
                        text="{:.1f}".format(results_biomeca[i][key]),
                        org=(int(x), int(biomechanicsImageInformation.dict[key][1])),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.6,
                        color=Drawing.color_white,
                        thickness=1,
                        lineType=cv2.LINE_AA)

        O, trunk_reference_system = PoseFeatures.compute_trunk_reference_system(left_shoulder=results_w_i[i][11],
                                                                                right_shoulder=results_w_i[i][12],
                                                                                left_hip=results_w_i[i][23],
                                                                                right_hip=results_w_i[i][24])

        image = DrawingUtilities.draw_reference_system(frame=image,
                                                       center=O,
                                                       reference_system=trunk_reference_system,
                                                       scale=50,
                                                       image_width=image_width,
                                                       image_height=image_height,
                                                       bio_info_dict=biomechanicsInformation.dict)



        image_list.append(image)

VideoUtilities.save_images_to_video(images=image_list,
                                    save_path=KentaPathInfo().saved_video_path + video_name + "_info.mp4",
                                    image_height=image_height,
                                    image_width=image_width)

# Set temporary folder for saving figures and making the video
image_folder_temp = os.getcwd() + "\\Temp\\"
if not os.path.exists(image_folder_temp):
    os.mkdir(image_folder_temp)

# loop in the frames got, save the Video
filenames = []
for i in range(n_frame):

    O, trunk_reference_system = PoseFeatures.compute_trunk_reference_system(left_shoulder=results_w[i][11],
                                                                            right_shoulder=results_w[i][12],
                                                                            left_hip=results_w[i][23],
                                                                            right_hip=results_w[i][24])

    mp_drawing.plot_landmarks(landmark_list=results[i].pose_world_landmarks, connections=mp_holistic.POSE_CONNECTIONS,
                              landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1,
                                                                           circle_radius=1),
                              connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2))

    xaxis = trunk_reference_system[:, 0].transpose()
    yaxis = trunk_reference_system[:, 1].transpose()
    zaxis = trunk_reference_system[:, 2].transpose()

    # fig = plt.figure(figsize=(8, 8))
    plt.tight_layout()
    fig = plt.gcf()
    ax = plt.gca()

    ax.scatter(O[0], O[1], O[2], color="black")

    _quiver(ax=ax,
            origin=O,
            vector=xaxis, color="red")
    _quiver(ax=ax,
            origin=O,
            vector=yaxis, color="green")
    _quiver(ax=ax,
            origin=O,
            vector=zaxis, color="blue")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D pose at frame: " + str(i) + "/" + str(n_frame))
    ax.set_xlim((-1, 1))
    ax.set_ylim((-1, 1))
    ax.set_zlim((-1, 1))

    angle = 50
    ax.view_init(30, angle)

    print("Drawing 3D pose at frame: " + str(i) + "/" + str(n_frame))
    filenames.append(str(i).zfill(6))
    fig.savefig(image_folder_temp + str(i).zfill(6) + ".png", dpi=100)
    plt.close(fig)
# For saving the video, generate imageio
with imageio.get_writer(KentaPathInfo.saved_video_path + video_name + "_3D_pose.mp4", mode="I") as writer:
    for filename in filenames:
        image = imageio.imread(image_folder_temp + filename + ".png")
        writer.append_data(image)
shutil.rmtree(image_folder_temp)

# Computing rate of change of angles
roll = []
roll_v = []
pitch = []
pitch_v = []
yaw = []
yaw_v = []
for i in range(n_frame):
    roll.append(results_biomeca[i]["trunk_roll"])
    pitch.append(results_biomeca[i]["trunk_pitch"])
    yaw.append(results_biomeca[i]["trunk_yaw"])

for i in range(n_frame - 1):
    roll_v.append((roll[i] - roll[i+1])/(1/fps))
    pitch_v.append((pitch[i] - pitch[i+1])/(1/fps))
    yaw_v.append((yaw[i] - yaw[i+1])/(1/fps))

fig = plt.figure(figsize=(18, 8))
ax = fig.add_subplot(1, 1, 1)
ax.plot(np.array(roll_v), label="roll")
ax.plot(np.array(pitch_v), label="pitch")
ax.plot(np.array(yaw_v), label="yaw")
ax.set_title("Trunk")
ax.set_xlabel("Frame")
ax.set_ylabel("Amount of angle change (angular velocity) [rad/sec]")
plt.legend(loc=2, frameon=True, fancybox=False, ncol=3, framealpha=0.5, edgecolor="black")
plt.show()


