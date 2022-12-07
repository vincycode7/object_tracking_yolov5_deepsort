import cv2, tempfile
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from detector_utils.detector import YoloObjectTrackerFrame
import pandas as pd

@st.cache
def load_model_detector():
    model =  YoloObjectTrackerFrame()
    return model
global deepsort_memory
deepsort_memory = None

def getAvaillableCams():
    # start from index one and loops till all cameras are picked
    max_cam_port = 10
    arr = []
    
    for each_index in range(max_cam_port):
        try:
            cap = cv2.VideoCapture(each_index)
            if cap.isOpened():
                arr.append(each_index)
            cap.release()
            cv2.destroyAllWindows()
        except:
            pass
    arr.append(None)
    return arr

# Input component
def process_input_feed(input_type, write_input_to_canvas, names=[],write_output_to_canvas=None, detector=None, perform_inference=False, confidence=0.40,iou=0.8, save_enc_img_feature=False):
    """
        This function is the input function, It accepts different inputs such as output canvas, input canvas
        media input type. It uses this information to request for input from the user, pass that to the 
        backend  object detector and then putting the outputs on the input and output canvas.

        input_type: Type of input to request from user.
        write_input_to_canvas: canvas to write user inputs.
        names: names of objects to detect from user inputs
        write_output_to_canvas:; canvas to write backend detector outputs.
        detector: The  backend dectector class responsible for the detectoin of objects. 
        perform_inference: If True it performs inference on inputs.
    """
    global deepsort_memory
    if deepsort_memory == None:
        deepsort_memory = detector._init_tracker(save_enc_img_feature=save_enc_img_feature)

    # deepsort_memory = [None, None]
    # Pick classes to use during detection
    # Variables Used to Calculate FPS
    prev_frame_time = 0 # Variables Used to Calculate FPS
    new_frame_time = 0
    filter_classes = st.sidebar.checkbox("Enable Custom Class Filter", True)
    picked_class_ids = []
    default_class = ["car","truck","motorcycle"]

    if filter_classes:
        picked_class = st.sidebar.multiselect("Select Classes to use for output", list(names), default=default_class)  
    else:
        picked_class = default_class
    
    for each in picked_class: 
        picked_class_ids.append(names.index(each))

    classes=picked_class_ids
    
    # Set default values
    default_img_path= "static_files/test_image.jpg" #"static_files/JPEG_20221017_084112_6779959420174114919.jpg" 
    default_vid_path="static_files/test_video.mp4"

    # Get inputs
    if input_type=="Image":
        image_file_buff = write_input_to_canvas.file_uploader("Upload an Image File", type=['png', 'jpg'])
        default_test_img = write_input_to_canvas.checkbox("Use Default Test Image", True)
    elif input_type in ["Video"]:
        video_file_buff = write_input_to_canvas.file_uploader("Upload a Video File", type=["mp4", "mov", "avi", "asf","m4v"])
        default_test_vid = write_input_to_canvas.checkbox("Use Default Test Video", True)        
    elif input_type in ["Camera"]:
        available_cams = getAvaillableCams()
        cam_index = write_input_to_canvas.selectbox("Select Camera", available_cams)

    inputLocationImg = write_output_to_canvas.sidebar.empty()
    inputLocationImg.image([])
    outputLocation = write_output_to_canvas.empty()
    outputDataframeLocation = write_output_to_canvas.empty()
    display_input_file = st.sidebar.checkbox("Show Input", False)
    demacateLocation = write_output_to_canvas.sidebar.empty()

    # Run detection and display 
    if input_type=="Image":
        deepsort_memory = detector._init_tracker(save_enc_img_feature=save_enc_img_feature)
        # inputLocationImg = write_output_to_canvas.sidebar.empty()
        inputLocationImg.image([])

        if image_file_buff or default_test_img:
            if image_file_buff:
                img = Image.open(image_file_buff)  
            else:
                img = Image.open(default_img_path)  
            img = np.asarray(img)
            image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if isinstance(type(image), type(None)):
                image = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
              
            if perform_inference and detector:
                image_with_boxes,deepsort_memory, detection_time, tracking_time = detector.image_dectection(image, classes=classes, conf_thres=confidence, draw_box_on_img=True, iou_thres=iou, deepsort_memory=deepsort_memory)
                image_with_boxes = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
                if display_input_file:
                    inputLocationImg.image(img)    
                outputLocation.image(image_with_boxes)
                try:
                    data = pd.DataFrame(deepsort_memory.results["class_metric"])
                    data_fields = outputDataframeLocation.columns(2)
                    data_fields[0].dataframe(data=data.loc['class_count'])
                    data_fields[1].dataframe(data=data.loc['location_unique_id'])
                except:
                    pass

    if input_type in ["Video", "Camera"] and perform_inference:
        file_name = None
        if input_type in ["Video"]:
            if video_file_buff:
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(video_file_buff.read())
                file_name = tfile.name
            elif default_test_vid:
                file_name = default_vid_path
        if input_type in ["Camera"]:
            file_name = int(cam_index) if cam_index != None else None

        # Run detection
        if file_name != None:
            deepsort_memory = detector._init_tracker(save_enc_img_feature=save_enc_img_feature)
            # inputLocationImg = write_output_to_canvas.sidebar.empty()
            multi_input = cv2.VideoCapture(file_name)

            while multi_input.isOpened():
                check, frame = multi_input.read()
                if check:
                    # cv2.imshow("Image", frame)
                    # multi_input.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    if perform_inference and detector:
                        image = np.asarray(frame)
                        image_with_boxes,deepsort_memory, detection_time, tracking_time = detector.image_dectection(image, classes=classes, conf_thres=confidence, draw_box_on_img=True, iou_thres=iou, deepsort_memory=deepsort_memory)
                        image_with_boxes = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
                        if display_input_file:
                            inputLocationImg.image(frame2)
                        outputLocation.image(image_with_boxes)

                        try:
                            data = pd.DataFrame(deepsort_memory.results["class_metric"])
                            data_fields = outputDataframeLocation.columns(2)
                            data_fields[0].dataframe(data=data.loc['class_count'])
                            data_fields[1].dataframe(data=data.loc['location_unique_id'])
                        except:
                            pass
                else:
                    outputLocation.write('No video')
            
            # Closes all the frames
            cv2.destroyAllWindows()

    demacateLocation.markdown("---")
            
def inference():

    """
        This function is responsible for getting user input from frontend, passing it to
        the backend functions and methods and then writing the result back to the frontend.
    """
    # st.markdown("#Detection and Tracking with YOLOv5s")
    st.markdown("##### Instruction: Set your configuration on the left sidebar panel then toggle the `Run Solution` checkbox below to get started.")
    st.markdown("###### **`Please note, If you are getting a low detection / tracking rate, try changing the confidence threshold on the left side bar.`**")

    col1, col2, col3 = st.columns(3)
    perform_inference = col2.checkbox("Run Solution", False)

    Solution_state = "Running" if perform_inference else "Not Running"
    st.markdown(f"###### Status: {Solution_state}.")
    st.sidebar.markdown(f"## Configuration")
    st.sidebar.markdown("---")
    confidence = st.sidebar.slider("""**Prediction Confidence Threshold** [This threshold specifies the accepted probability value of a box belonging to a class.]""", min_value=0.0, max_value=1.0, value=0.35)
    iou = st.sidebar.slider("""**Intersection Over Union Threshold** [This threshold specifies the accepted probability value of overlap between two detected bounding boxes.]""", min_value=0.0, max_value=1.0, value=0.30)
    st.sidebar.markdown("---") 

    # Checkbox configuration
    # use_optimized = st.sidebar.checkbox("Use Optimized Model")
    save_enc_img_feature = st.sidebar.checkbox("Save Encoded Feature Outputs")
    # run_on_gpu = st.sidebar.checkbox("Run on GPU")

    # st.sidebar.markdown("---")

    input_type = st.sidebar.selectbox("Input Type", ["Image","Video", "Camera"])

    # Get model
    model = load_model_detector()
    names = list(model.key_to_string.values())

    process_input_feed(write_input_to_canvas=st.sidebar, input_type=input_type, write_output_to_canvas=st, names=names, detector=model, confidence=confidence, perform_inference=perform_inference,iou=iou,save_enc_img_feature=save_enc_img_feature)
    perform_inference = False

    # Display input and output data
    st.markdown("###### - **The `Class Counts` are the unique object counts across frames without duplicates.**")
    st.markdown("###### - **The `Location Unique Ids`, is a one-on-one between values assigned to unique objects and values used for representation on the image, deepsort values as key while representation value as value.**")


# inference()