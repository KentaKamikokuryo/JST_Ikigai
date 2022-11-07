import cv2
import mediapipe as mp
from matplotlib import cm
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
import numpy as np

class Index:

    # Around the face
    index_face_border = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
                         176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

    # Eye edge loop
    index_eye_edge_loop_right_1 = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
    index_eye_edge_loop_right_2 = [130, 247, 30, 29, 27, 28, 56, 190, 243, 112, 26, 22, 23, 24, 110, 25]
    index_eye_edge_loop_right_3 = [226, 113, 225, 224, 223, 222, 221, 189, 244, 233, 232, 231, 230, 229, 228, 31]

    index_eye_edge_loop_left_1 = [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]
    index_eye_edge_loop_left_2 = [359, 467, 260, 259, 257, 258, 286, 414, 463, 341, 256, 252, 253, 254, 339, 255]
    index_eye_edge_loop_left_3 = [446, 342, 445, 444, 443, 442, 441, 413, 464, 453, 452, 451, 450, 449, 448, 261]

    # Iris
    index_left_iris = [474, 475, 476, 477]
    index_right_iris = [469, 470, 471, 472]
