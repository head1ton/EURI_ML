from ultralytics import YOLO
import cv2

model_path = 'best.pt'

image_path = 'tiger.jpg'

img = cv2.imread(image_path)

H, W, _ = img.shape
# print(img.shape)

model = YOLO(model_path)
# model = YOLO("yolov8n-seg.pt")

results = model(img)
print(results)

for result in results:
    for j, mask in enumerate(result.masks.data):
        mask = mask.numpy() * 255
        mask = cv2.resize(mask, (W, H))

        cv2.imwrite("./output.png", mask)