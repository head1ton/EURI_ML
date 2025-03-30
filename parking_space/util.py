import pickle

from skimage.transform import resize
import numpy as np
import cv2

EMPTY = True
NOT_EMPTY = False

MODEL = pickle.load(open("./model/model.p", "rb"))

def empty_or_not(spot_bgr):
    flat_data = []

    img_resized = resize(spot_bgr, (15, 15, 3))
    flat_data.append(img_resized.flatten())
    flat_data = np.array(flat_data)

    y_output = MODEL.predict(flat_data)

    if y_output == 0:
        return EMPTY
    else:
        return NOT_EMPTY

def get_parking_spots_bboxes(connected_components):

    (ret_val, labels, stats, centroid) = connected_components

    slots = []
    coef = 1
    for i in range(1, ret_val):
        x1 = int(stats[i, cv2.CC_STAT_LEFT] * coef)
        y1 = int(stats[i, cv2.CC_STAT_TOP] * coef)
        w = int(stats[i, cv2.CC_STAT_WIDTH] * coef)
        h = int(stats[i, cv2.CC_STAT_HEIGHT] * coef)

        slots.append([x1, y1, w, h])

    return slots
