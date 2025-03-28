from ultralytics import solutions
# solutions.ParkingPtsSelection()

import cv2

polygon_json_path = "bounding_box.json"

cap = cv2.VideoCapture("smart_parking.mp4")
assert cap.isOpened(), "Error reading video file"

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

video_writer = cv2.VideoWriter("parking_management.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

management = solutions.ParkingManagement(model_path="yolov8n.pt")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    json_data = management.parking_regions_extraction(polygon_json_path)
    results = management.model.track(frame, persist=True, show=False)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().tolist()
        clss = results[0].boxes.clss.cpu().tolist()
        management.process_data(json_data, frame, boxes, clss)

    management.display_frames(frame)
    video_writer.write(frame)

cap.release()
video_writer.release()
cv2.destroyAllWindows()
