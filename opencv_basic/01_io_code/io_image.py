import os
import cv2

image_path = os.path.join('..', 'data', 'lion.jpg')

img = cv2.imread(image_path)

cv2.imshow(os.path.join('..', 'data', 'lion_out.jpg'), img)

cv2.imshow('image', img)

cv2.waitKey(0)