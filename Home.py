#Todos:
# Implement Image upload using streamlit (Done)
# Implement Object detection on streamlit Image (Done)

# Implement Video upload using streamlit (Done)
# Implement Object detection on streamlit Video (Done)
# Implement Object detection, tracking and counting  on streamlit Video (Done)
# Implement FPS metrics (Done)
# Prevent yolov5 from always connecting to internet after weight is downloaded(Done)
# Prevent warnings from getting printed in terminal for opencv(Done)
# Update Readme to reflect new UI (done)

# Update docker to run on render
# Add extra test on new push
# Use git action to run test

import cv2, tempfile, pipes, os
import streamlit as st
import json
from detector_utils.inference import inference
# from pages.inference import inference
# with open("static_files/yolo_classes.txt") as f:
#     lines = f.readlines()
# new_lines = { int(item.split(":")[0]) : item.split(":")[1].strip(" ").strip("\n") for item in lines}
# print("export OPENCV_LOG_LEVEL=OFF" % (pipes.quote(str("OFF")))) #disable opencv logging
def main():
    st.set_page_config(
        page_title="Home",
)
    st.markdown("### Welcome to this Object Detection and Tracking Solution (YOLOv5s)", True)
    inference()

main()