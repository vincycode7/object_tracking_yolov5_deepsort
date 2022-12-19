import cv2, tempfile
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from detector_utils.detector import YoloObjectTrackerFrame
import pandas as pd
import time
import math

def hash_model_reference(model_reference):
    return (
        model_reference.kwargs, 
        model_reference.is_sparsed_optimisation)

# @st.cache
@st.cache(hash_funcs={YoloObjectTrackerFrame: hash_model_reference})
def load_optimised_model_detector(**kwargs):
    model =  YoloObjectTrackerFrame(**kwargs)
    return model

@st.cache
def load_base_model_detector(**kwargs):
    model =  YoloObjectTrackerFrame(**kwargs)
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
    if perform_inference and detector:
        global deepsort_memory
        deepsort_memory = detector._init_tracker(save_enc_img_feature=save_enc_img_feature)

    def write_output_single_frame(perform_inference, detector, frame, classes, confidence, draw_box_on_img, iou, deepsort_memory, old_detection_fps, display_input_file, inputLocationImg, outputLocation, outputFPS, outputDataframeLocations):
        if display_input_file:
                frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                inputLocationImg.image(frame2)
        if perform_inference and detector:
            image = np.asarray(frame)
            image_with_boxes,deepsort_memory, new_detection_fps = detector.image_dectection(image, classes=classes, conf_thres=confidence, draw_box_on_img=draw_box_on_img, iou_thres=iou, deepsort_memory=deepsort_memory)
            detection_fps = round((old_detection_fps+new_detection_fps)/2,2) if old_detection_fps > 0 else round(new_detection_fps,2)
            image_with_boxes = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
            image_with_boxes = cv2.resize(image_with_boxes, (800, 470), interpolation=cv2.INTER_LINEAR)
            outputLocation.image(image_with_boxes)
            col1, col2, col3 = outputFPS.columns(3)
            col1.metric(label="Detection FPS", value=str(detection_fps), delta=str(round(detection_fps-old_detection_fps,2)))

            try:
                data = pd.DataFrame(deepsort_memory.results["class_metric"])
                outputDataframeLocations[0].table(data=data.loc['class_count'])
                col2.metric(label="Total Classes Detected", value=str(data.loc['class_count'].shape[0]))
                outputDataframeLocations[1].table(data=data.loc['location_unique_id'].astype('str'))
                col3.metric(label="Total Objects Detected", value=str(sum(data.loc['class_count'])))
            except:
                pass
            return detection_fps
        return 0

    # global deepsort_memory
    # if deepsort_memory == None:
    #     deepsort_memory = detector._init_tracker(save_enc_img_feature=save_enc_img_feature)

    # deepsort_memory = [None, None]
    # Pick classes to use during detection
    # Variables Used to Calculate FPS
    old_detection_fps = 0 
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
    outputFPS = write_output_to_canvas.empty()
    outputDataframeLocation1 = write_output_to_canvas.empty()
    outputDataframeLocation2 = write_output_to_canvas.empty()
    display_input_file = st.sidebar.checkbox("Show Input", True)
    demacateLocation = write_output_to_canvas.sidebar.empty()

    # Run detection and display 
    if input_type=="Image":
        # deepsort_memory = detector._init_tracker(save_enc_img_feature=save_enc_img_feature)
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
            old_detection_fps = write_output_single_frame(perform_inference=perform_inference, detector=detector, frame=image, classes=classes, confidence=confidence, draw_box_on_img=True, iou=iou, deepsort_memory=deepsort_memory, old_detection_fps=old_detection_fps, display_input_file=display_input_file, inputLocationImg=inputLocationImg, outputLocation=outputLocation, outputFPS=outputFPS, outputDataframeLocations=[outputDataframeLocation1, outputDataframeLocation2])

    if input_type in ["Video", "Camera"]:
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
            # deepsort_memory = detector._init_tracker(save_enc_img_feature=save_enc_img_feature)
            # inputLocationImg = write_output_to_canvas.sidebar.empty()
            multi_input = cv2.VideoCapture(file_name)

            while multi_input.isOpened():
                check, frame = multi_input.read()
                if check:
                    # cv2.imshow("Image", frame)
                    # multi_input.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    old_detection_fps = write_output_single_frame(perform_inference=perform_inference, detector=detector, frame=frame, classes=classes, confidence=confidence, draw_box_on_img=True, iou=iou, deepsort_memory=deepsort_memory, old_detection_fps=old_detection_fps, display_input_file=display_input_file, inputLocationImg=inputLocationImg, outputLocation=outputLocation, outputFPS=outputFPS, outputDataframeLocations=[outputDataframeLocation1, outputDataframeLocation2])

                else:
                    outputLocation.write('No Frame')
            
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
    emptyButton = col2.empty()
    perform_inference = False
    stopButton = False

    if not perform_inference or stopButton:
        startButton = emptyButton.button("Click Here To Start Inference")
    if startButton:
        perform_inference = True
        stopButton = emptyButton.button("Click Here To Stop Inference")

    Solution_state = "Running" if perform_inference else "Not Running"
    runningStatus = st.empty()
    runningStatus.markdown(f"###### Status: {Solution_state}.")
    st.sidebar.markdown(f"## Configuration")
    st.sidebar.markdown("---")
    is_sparsed_optimisation = st.sidebar.checkbox("Enable Model Optimisation (This will use deepspare's prunned and quantised model)", True)
    st.sidebar.markdown("---")

    if not is_sparsed_optimisation:
        model_size = st.sidebar.selectbox("""**Yolo Size Ultralytics** - [Click Here](https://pytorch.org/hub/ultralytics_yolov5/#model-description) or [Click Here](https://github.com/ultralytics/yolov5/#pretrained-checkpoints) for more information on model size. Do note, initial weight download might take a while depending on your internet speed and model size specified.""", ['yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x'])
    else:
        model_size = st.sidebar.selectbox("""**Yolo Size DeepSparse** - [Click Here](https://github.com/neuralmagic/sparseml/blob/main/integrations/ultralytics-yolov5/tutorials/sparsifying_yolov5_using_recipes.md#applying-a-recipe) for more information on model size. Do note, initial weight download might take a while depending on your internet speed and model size specified.""", ['yolov5s-pq','yolov5s-p', 'yolov5l-pq'])
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
    if not is_sparsed_optimisation:
        model = load_base_model_detector(model_size=model_size,is_sparsed_optimisation=is_sparsed_optimisation)
    else:
        model = load_optimised_model_detector(model_size=model_size,is_sparsed_optimisation=is_sparsed_optimisation)

    names = list(model.key_to_string.values())

    process_input_feed(write_input_to_canvas=st.sidebar, input_type=input_type, write_output_to_canvas=st, names=names, detector=model, confidence=confidence, perform_inference=perform_inference,iou=iou,save_enc_img_feature=save_enc_img_feature)

    if perform_inference:
        resetButton = emptyButton.button("Click Here To Reset Inference State")
        if resetButton:
            perform_inference = False
            stopButton = False
            Solution_state = "Running" if perform_inference else "Not Running"

    # runningStatus.markdown(f"###### Status: {Solution_state}.")

    # Display input and output data
    st.markdown("###### - **The `Class Counts` are the unique object counts across frames without duplicates.**")
    st.markdown("###### - **The `Location Unique Ids`, is a one-on-one between values assigned to unique objects and values used for representation on the image, deepsort values as key while representation value as value.**")


# inference()