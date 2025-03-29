from ultralytics import YOLO
import cv2

model_path = 'best.pt'

image_path = 'horse3.jpg'

img = cv2.imread(image_path)

model = YOLO(model_path)

results = model(image_path)[0]
# print('results : ', results)

for result in results:
    # print(result.keypoints)
    # print(len(result.keypoints.data.tolist()[0]))
    for keypoint_index, keypoint in enumerate(result.keypoints.data.tolist()[0]):
        # print(keypoint[0][keypoint_index], keypoint[0][keypoint_index])
        # print(keypoint_index)
        cv2.putText(img, str(keypoint_index), (int(keypoint[0]), int(keypoint[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imshow('img', img)
cv2.waitKey(0)

