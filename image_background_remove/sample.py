import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates as im_coordinates

# 이미지 업로드
file = st.file_uploader('Input Image', type=['jpeg', 'jpg', 'png'])

if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, caption='Uploaded Image')

    # 이미지 좌표 선택
    value = im_coordinates(image)
    if value is not None:
        st.write(f"Selected coordinates: {value}")
    else:
        st.write("No coordinates selected")

    # 디버깅을 위한 로그 출력
    st.write(f"Image size: {image.size}")
    st.write(f"File name: {file.name}")
    st.write(f"Image mode: {image.mode}")
    st.write(f"Image format: {image.format}")
    st.write(f"Image info: {image.info}")