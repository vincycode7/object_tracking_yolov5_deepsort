# Implement Image upload using streamlit
# Implement Object detection on streamlit Image

# Implement Video upload using streamlit
# Implement Object detection on streamlit Video
#Implement Object detection, tracking and counting  on streamlit Video

import cv2, tempfile
import streamlit as st
import json
from detector_utils.inference import inference
# from pages.inference import inference
# with open("static_files/yolo_classes.txt") as f:
#     lines = f.readlines()
# new_lines = { int(item.split(":")[0]) : item.split(":")[1].strip(" ").strip("\n") for item in lines}

def main():
    st.set_page_config(
        page_title="Home",
)
    st.markdown("### Welcome to InstaDeep Detection and Tracking Presentation (YOLOv5s)", True)
    inference()

main()