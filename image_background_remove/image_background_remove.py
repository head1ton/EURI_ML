import base64
import os
import time

import requests
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates as im_coordinates
import cv2
import numpy as np
from streamlit_dimensions import st_dimensions


st.set_page_config(layout='wide')

def set_background(image_file):
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
            body {{
                background-image: url('data:image/png;base64,{b64_encoded}');
                background-size: cover;
            }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

set_background('background.jpg')

api_endpoint = "https://rick.us-east-2.aws.modelbit.com/v1/remove_background/latest"

col01, col02 = st.columns(2)

file = col02.file_uploader('Input Image', type=['jpeg', 'jpg', 'png'], label_visibility='collapsed')

if file is not None:
    image = Image.open(file).convert('RGB')
    # print(image)
    screen_dim = st_dimensions(key='main')
    # print(screen_dim['width'])

    width = 100
    if screen_dim is not None:
        width = int(screen_dim['width'] / 2)

    image = image.resize((width, int(image.height * width / image.width )))

    col1, col2 = col02.columns(2)

    placeholder0 = col02.empty()
    with placeholder0:
        value = im_coordinates(image)
        print('value : ', value)
        if value is not None:
            print(value)

    if col1.button('Original Image', use_container_width=True):
        placeholder0.empty()
        placeholder1 = col02.empty()
        with placeholder1:
            col02.image(image, use_column_width=True)

    if col2.button('Remove Background Image', type='primary', use_container_width=True):
        placeholder0.empty()
        placeholder2 = col02.empty()

        filename = '{}_{}_{}.png'.format(file.name, value['x'], value['y'])

        if os.path.exists(filename):
            result_image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        else:
            _, image_bytes = cv2.imencode('.png', np.asarray(image))

            image_bytes = image_bytes.tobytes()

            image_bytes_encoded_base64 = base64.b64encode(image_bytes).decode('utf-8')

            api_data = {'data': [image_bytes_encoded_base64, value['x'], value['y']]}
            response = requests.post(api_endpoint, json=api_data)
            result_image = response.json()['data']

            result_image_bytes = base64.b64decode(result_image)

            result_image = cv2.imdecode(np.frombuffer(result_image_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

            cv2.imwrite(filename, result_image)

        with placeholder2:
            col02.image(result_image, use_column_width=True)