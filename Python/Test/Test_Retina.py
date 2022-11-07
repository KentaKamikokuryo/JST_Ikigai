from retinaface import RetinaFace
import cv2
from Classes.PathInfo import PathInfo

path_info = PathInfo()

img_path = path_info.path_data_test + "test.jpg"
img = cv2.imread(img_path)

resp = RetinaFace.detect_faces(img_path, threshold=0.5)
print("faces:" + str(len(resp)))


def int_tuple(t):
    return tuple(int(x) for x in t)


for key in resp:
    identity = resp[key]

    # ---------------------

    landmarks = identity["landmarks"]
    diameter = 1
    cv2.circle(img, int_tuple(landmarks["left_eye"]), diameter, (0, 0, 255), -1)
    cv2.circle(img, int_tuple(landmarks["right_eye"]), diameter, (0, 0, 255), -1)
    cv2.circle(img, int_tuple(landmarks["nose"]), diameter, (0, 0, 255), -1)
    cv2.circle(img, int_tuple(landmarks["mouth_right"]), diameter, (0, 0, 255), -1)

    facial_area = identity["facial_area"]
    cv2.rectangle(img, (facial_area[2], facial_area[3]), (facial_area[0], facial_area[1]), (255, 255, 255), 1)
    # facial_img = img[facial_area[1]: facial_area[3], facial_area[0]: facial_area[2]]
    # plt.imshow(facial_img[:, :, ::-1])

cv2.imwrite('output.' + img_path.split(".")[1], img)
