import os
import shutil

import boto3
import cv2
import matplotlib.pyplot as plt

import credentials
from aws_rekognition.main import reko_client

output_dir = "./output_recording"
anns_dir = os.path.join(output_dir, "anns")
imgs_dir = os.path.join(output_dir, "imgs")

for dir_ in [output_dir, anns_dir, imgs_dir]:
    if os.path.exists(dir_):
        shutil.rmtree(dir_)
    os.mkdir(dir_)

reko_client = boto3.client('rekognition',
                           aws_access_key_id=credentials.access_key,
                           aws_secret_access_key=credentials.secret_key)

input_file = "test.mp4"
cap = cv2.VideoCapture(input_file)

counter = 0

class_names = []

while True:
    cap.set(cv2.CAP_PROP_POS_FRAMES, counter)
    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape

    anns_file = open(os.path.join(anns_dir, '{}.txt'.format(str(counter).zfill(6))), 'w')

    tmp_filename = './tmp.jpg'
    cv2.imwrite(tmp_filename, frame)
    cv2.imwrite(os.path.join(imgs_dir, '{}.jpg'.format(str(counter).zfill(6))), frame)

    with open(tmp_filename, 'rb') as image:
        response = reko_client.detect_labels(Image={'Bytes': image.read()})

    for label in response['Labels']:
        if len(label['Instances']) > 0:
            name = label['Name']
            if name not in class_names:
                class_names.append(name)
            for instance in response['Instances']:
                conf = float(instance['Confidence']) / 100
                w = instance['BoundingBox']['Width']
                h = instance['BoundingBox']['Height']
                x = instance['BoundingBox']['Left']
                y = instance['BoundingBox']['Top']

                anns_file.write('{} {} {} {} {} {} {}\n'.format(class_names.index(name),
                                                                x + (w / 2),
                                                                y + (h / 2),
                                                                w,
                                                                h,
                                                                conf))

    os.remove(tmp_filename)

    anns_file.close()

    counter += 1

with open(os.path.join(output_dir, 'class.names'), 'w') as fw:
    for name in class_names:
        fw.write('{}\n'.format(name))
    fw.close()
