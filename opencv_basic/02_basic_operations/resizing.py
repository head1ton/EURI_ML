import os
import cv2

img = cv2.imread(os.path.join('..', 'data', 'lion.jpg'))
# (853, 1280, 3)
resized_img = cv2.resize(img, (453, 640))

print(img.shape)
print(resized_img.shape)

cv2.imshow('img', img)
cv2.imshow('resized_img', resized_img)

cv2.waitKey(0)