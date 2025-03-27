import base64
import os

import requests
import streamlit as st
from PIL import Image
from streamlit_dimensions import st_dimensions
from streamlit_image_coordinates import streamlit_image_coordinates as im_coordinates
import cv2
import numpy as np


st.set_page_config(layout='wide')

def set_background(image_file):
    """
    This function sets the background of a Streamlit app to an image specified by the given image file.

    Parameters:
        image_file (str): The path to the image file to be used as the background.

    Returns:
        None
    """
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)


set_background('./background.jpg')

api_endpoint = "https://rick.us-east-2.aws.modelbit.com/v1/remove_background/latest"

image_folder = './images'

os.makedirs(image_folder, exist_ok=True)

col01, col02 = st.columns(2)

# file uploader
file = col02.file_uploader('image', type=['jpeg', 'jpg', 'png'], label_visibility='hidden')

# read image
if file is not None:
    image = Image.open(file).convert('RGB')

    screen_dim = st_dimensions(key='main')
    width = 100
    if screen_dim is not None:
        width = int(screen_dim['width'] // 2)

    image = image.resize((width, int(image.height * width / image.width)))

    # create buttons
    col1, col2 = col02.columns(2)

    # visualize image
    # click on image, get coordinates
    placeholder0 = col02.empty()
    with placeholder0:
        value = im_coordinates(image, key="pil")
        if value is not None:
            print(value)

    if col1.button('Original', use_container_width=True):
        placeholder0.empty()
        placeholder1 = col02.empty()
        with placeholder1:
            col02.image(image, use_container_width=True)

    if col2.button('Remove background', type='primary', use_container_width=True):
        # call api
        placeholder0.empty()
        placeholder2 = col02.empty()

        # filename = '{}_{}_{}.png'.format(file.name, value['x'], value['y'])
        filename = '{}_{}_{}.png'.format(file.name.split('.')[0], int(image.width // 2), int(image.height // 2))

        if os.path.exists(os.path.join(image_folder, filename)):
            result_image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        else:
            _, image_bytes = cv2.imencode('.png', np.asarray(image))

            image_bytes = image_bytes.tobytes()

            image_bytes_encoded_base64 = base64.b64encode(image_bytes).decode('utf-8')

            # api_data = {"data": [image_bytes_encoded_base64, value['x'], value['y']]}
            api_data = {"data": [image_bytes_encoded_base64, int(image.width // 2), int(image.height // 2)]}
            response = requests.post(api_endpoint, json=api_data)

            result_image = response.json()['data']

            result_image_bytes = base64.b64decode(result_image)

            result_image = cv2.imdecode(np.frombuffer(result_image_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

            cv2.imwrite(os.path.join(image_folder, filename), result_image)


        with placeholder2:
            col02.image(result_image, use_container_width=True)