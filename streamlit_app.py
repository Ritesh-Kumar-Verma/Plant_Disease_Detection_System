import streamlit as st

st.title("🎈 AI - Plant Disease Detection")

st.write(
    "Upload image and get information about the disease"
)

image = st.file_uploader("Upload Image")


