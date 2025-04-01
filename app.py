import streamlit as st
import cv2
import face_recognition
import numpy as np
from PIL import Image

def load_image(image_file):
    image = Image.open(image_file)
    return np.array(image)

def encode_face(image):
    face_encodings = face_recognition.face_encodings(image)
    if len(face_encodings) > 0:
        return face_encodings[0]
    return None

st.title("Face Similarity App")

st.sidebar.header("Upload Images")
reference_image_file = st.sidebar.file_uploader("Upload Reference Image", type=["jpg", "png", "jpeg"])
compare_image_file = st.sidebar.file_uploader("Upload Image to Compare", type=["jpg", "png", "jpeg"])

if reference_image_file and compare_image_file:
    ref_img = load_image(reference_image_file)
    comp_img = load_image(compare_image_file)

    st.image([ref_img, comp_img], caption=["Reference Image", "Comparison Image"], width=300)

    ref_enc = encode_face(ref_img)
    comp_enc = encode_face(comp_img)

    if ref_enc is not None and comp_enc is not None:
        face_distance = face_recognition.face_distance([ref_enc], comp_enc)[0]
        similarity_score = 1 - face_distance

        st.subheader("Similarity Score: {:.2f}".format(similarity_score))
        if similarity_score > 0.6:
            st.success("Faces Match!")
        else:
            st.error("Faces Do Not Match")
    else:
        st.error("No faces detected in one or both images.")
