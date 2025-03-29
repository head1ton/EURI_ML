import os
import boto3
import cv2
import credentials

output_dir = "./data"
output_dir_imgs = os.path.join(output_dir, 'imgs')
output_dir_anns = os.path.join(output_dir, 'anns')

reko_client = boto3.client('rekognition',
                           aws_access_key_id=credentials.access_key,
                           aws_secret_access_key=credentials.secret_key)

target_class = 'Zebra'

cap = cv2.VideoCapture('./movies.mp4')

frame_nmr = -1

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame_nmr += 1
    H, W, _ = frame.shape

    _, buffer = cv2.imencode('jpg', frame)

    image_bytes = buffer.tobytes()

    response = reko_client.detect_labels(Image={'Bytes':image_bytes}, MinConfidence=50)

    with open(os.path.join(output_dir_anns, 'frame_{}.txt'.format(str(frame_nmr).zfill(6))), 'w') as f:
        for label in response['Labels']:
            if label['Name'] == target_class:
                for instance_nmr in range(len(label['Instances'])):
                    bbox = label['Instances'][instance_nmr]['BoundingBox']
                    x1 = bbox['Left']
                    y1 = bbox['Top']
                    width = bbox['Width']
                    height = bbox['Height']

                    f.write('{} {} {} {} {}\n'.format(0,
                                                      (x1 + width / 2),
                                                      (y1 + height / 2),
                                                      width,
                                                      height))
        f.close()

    cv2.imwrite(os.path.join(output_dir_imgs, 'frame_{}.jpg'.format(str(frame_nmr).zfill(6))), frame)
