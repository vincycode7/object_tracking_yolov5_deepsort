## General System overview

![General System overview](./static_files/General%20System%20Architecture.png)

This is a solution to address the edge detection and tracking of specific objects for an imaginary company X, The general system architecture highlights the movement of optimized ml model downloaded from the server on to the edge device while inference is done on the edge device, It also highlights cases where data feedback are gotten to further train model. This process of data feedback to the server can pose a serious privacy issue, it is the duty of the ml engineers to prevent such privacy issues in their system implementation by using different masking techniques on the data before sending to the server. In this solution, you will be provided with option to make inference using optimised model(model that have been pruned or quantised) or base model(model that have not been pruned and quantised).

## An High level description of system implementation on an edge device.

![alt text](./static_files/Edge%20Device%20Architecture.png)

How does this solution work. Basically, Image, Video or Camera Inputs are gotten from a streamlit frontend which is then fed into a pytorch Yolo backend which then resize the input if necessary, convert color channel from BGR to RGB and then passes processed input through yolo model, Once this is completed, the result is then fed into a deepsort tracker which extras each image feature for each detected object, assigns integer values to each of the detected object and then tracks each detected object across frame. The final output is then sent back to the UI for Display.

## Mount Codebase to local workspace.

[Terminal-Version](Option 1)
   1. Open a `terminal` or `command prompt` 
   2. Run `git clone https://github.com/vincycode7/object_tracking_yolov5_deepsort.git`
   3. Once download is complete, change directory into the current working directory using `cd object_tracking_yolov5_deepsort/` as this directory contains all the code you need.
   3. Run `git submodule init`
   4. Run `git submodule update`
 
[Zip-version](Option 2)
   1. Navigate to the `<> Code` Dropdown
   2. Click to view options available
   3. Click on the download dro
   pdown to download a zipped file of all the code.
   4. Once download is complete, Unzip into a folder and open the parent folder `object_tracking_yolov5_deepsort` with any `IDE` of your choice.
   5. Run `git submodule init`
   6. Run `git submodule update`


## Dependency Installation and Running Project (Option 1 - Using Docker)
  1. Install docker locally
  2. Build a docker image using the `Dockerfile` by running `sudo docker build -t yolodeepsort:lastest .` or `docker build -t yolodeepsort:lastest .`
  3. Paste this in terminal, `sudo docker run -v ${PWD}:/app --device=/dev/video0 -p 80:80 yolodeepsort:lastest`.
  4. Note, with you get any error relation to `${PWD}` simply replace it with the abolute path to you current working directory by running `pwd` in terminal to get the absolute path of your current working directory
  and then replace `${PWD}` with the copy of the output from step `3` to run docker.
  5. Note, if you are having issues relating to the `--device=/dev/video0` you can simply remove it to disale webcam inferencing. Lastly you can specify `--device=path_to_device` multiple times if you have multiple web cams and they will be automatically displayed in UI, where `path_to_device` is the path where your device is located on your local machine.
  6. lastly once program is running, you can go to the url `http://localhost` or `http://0.0.0.0:80`

## Dependency Installation and Running Project (Option 2 - Installing dependencies manually)

#### Step 1 Install python

  1. Install python3
    - visit `https://code.visualstudio.com/docs/python/python-tutorial` for instructions

#### Step 2 Install pip

  2. Install pip
    
    [macOS]
      - Install pip - `python -m ensurepip --upgrade`
    
    [Ubuntu]
      - Install pip - `sudo apt-get install python3-pip`
    
    [Windows]
      - Visit `https://packaging.python.org/en/latest/tutorials/installing-packages/`

#### Step 3 Install pipenv

  3. Install pipenv
    
    [macOS]
      - Install pipenv `pip3 install pipenv`
      - Install xcode command line tool - `xcode-select --install`

    [Ubuntu]
      - Install pipenv - `pip3 install pipenv` or `sudo apt install pipenv`

    [Windows]
      - Install pipenv -  `pip install pipenv`

#### Step 4 - Create and activate an Environment for project

  1. Navigate to your project folder `cd object_tracking_yolov5_deepsort`
  2. Create a new Pipenv environment in that folder and activate that environment - `pipenv shell`

#### Step 5 - Project Dependencies Installations
  1. install dependencies using  `pipenv run pip3 install -r requirements.txt`

#### Step 6 - Running Project
  1. Confirm you are in your current working directory - `object_tracking_yolov5_deepsort`
  2. Activate the environment using the command  `pipenv shell`
  3. Run `export OPENCV_LOG_LEVEL=OFF`
  4. Run the app using the command `streamlit run Home.py --server.address=localhost` or `reset && streamlit run Home.py --server.address=localhost`


## General Note

Once project starts running you will have a web interface pop-up, In case you do not see a pop-up check you terminal to see the  localhost and port in this format    
`Local URL: http://localhost:port or Network URL: http://0.0.0.0:port` copy any one of them and paste in your browser. Once ui is loaded, ui will load model unto backend upon first interaction.

Please Note: Upon first run, project will download model weight which would then be saved in this path `./model_weight`, also note that this download process might take few seconds.

![UI Message while model is downloading model checkpoint of loading from previously downloaded checkpoint](./static_files/Screenshot%20from%202022-12-17%2003-20-13.png)


Once UI is loaded and model is initialized, Click on the center button to run or stop the program.
This is an example of what to expect when project starts running. Do note, that by the default the run checkbox is turned off, you have to hit the `Click Here  To Start Inference` Button to get the prediction running. Also you have option to test with default image and video  files, load your prefered image and video file or just run the solution in real time, using attached web cam if available.

![Start Inference](./static_files/Screenshot%20from%202022-12-17%2003-21-24.png)

![Inference Output](./static_files/Screenshot%20from%202022-12-17%2003-22-41.png)


## Interacting with the User Interface(UI).
  1. On the UI you will see difference configuration options on the left panel to help set thresholds and load input data, do take a look at all available options before going to item `2` below.
  2. Once you have viewed all available options, Toggle the `Run Solution` checkbox to either run or stop running the solution.
  3. While program is running, you should see bounding box information displayed on the image, also tracking information are displayed below the image.

![configuration1](./static_files/Screenshot%20from%202022-12-17%2003-23-04.png) ![configuration2](./static_files/Screenshot%20from%202022-12-17%2003-23-12.png) ![configuration3](./static_files/Screenshot%20from%202022-12-17%2003-23-21.png)

## Reference
1) [object_tracking_yolov5_deepsort](https://github.com/vincycode7/object_tracking_yolov5_deepsort)
2) [Yolov5_DeepSort_Pytorch](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch)   
3) [yolov5](https://github.com/ultralytics/yolov5)  
4) [deep_sort_pytorch](https://github.com/ZQPei/deep_sort_pytorch)       
5) [deep_sort](https://github.com/nwojke/deep_sort)   

Note: please follow the [LICENCE](https://github.com/ultralytics/yolov5/blob/master/LICENSE) of YOLOv5! 

Test Video by <a href="https://pixabay.com/users/mikes-photography-1860391/?utm_source=link-attribution&amp;utm_medium=referral&amp;utm_campaign=video&amp;utm_content=2165">Mikes-Photography</a> from <a href="https://pixabay.com//?utm_source=link-attribution&amp;utm_medium=referral&amp;utm_campaign=video&amp;utm_content=2165">Pixabay</a>

# Thank you !!!
