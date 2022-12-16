# Implement Image upload using streamlit (Done)
# Implement Object detection on streamlit Image (Done)

# Implement Video upload using streamlit (Done)
# Implement Object detection on streamlit Video (Done)
# Implement Object detection, tracking and counting  on streamlit Video (Done)
# Implement FPS metrics
# Prevent warnings from getting printed in terminal for opencv
# Look more into the to_cuda option

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
    st.markdown("### Welcome to this Object Detection and Tracking Solution (YOLOv5s)", True)
    inference()

main()