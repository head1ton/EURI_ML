import os
import cv2

def mouse_event(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f'x: {x}, y: {y}')


img = cv2.imread(os.path.join('..', 'data', 'lion.jpg'))

print(img.shape)

cropped_img = img[220:740, 320:940] # [y:y1, x:x1]

cv2.imshow('img', img)
cv2.imshow('cropped_img', cropped_img)

cv2.setMouseCallback('img', mouse_event, [img])

cv2.waitKey(0)