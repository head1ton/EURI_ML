import streamlit as st
import tensorflow as tf
from keras.layers import BatchNormalization, DepthwiseConv2D
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

from utils import classify, set_background

set_background('./background_hs.jpg')

st.title('Pneumonia classification')

st.header('Please upload a chest X-Ray image')

file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

custom_objects = {
    'DepthwiseConv2D': lambda **kwargs: DepthwiseConv2D(**{k: v for k, v in kwargs.items() if k != 'groups'}),
    'BatchNormalization': BatchNormalization
}
model = load_model('./model/pneumonia_classifier.h5', custom_objects=custom_objects)

with open('./model/labels.txt', 'r') as f:
    class_names = [a.replace('\n', '').split(' ')[1] for a in f.readlines()]
    f.close()

if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_container_width=True)

    class_name, conf_score = classify(image, model, class_names)

    st.write("### {}".format(class_name))
    st.write("### score: {}%".format(int(conf_score * 1000) / 10))