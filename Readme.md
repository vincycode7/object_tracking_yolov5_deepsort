


## Step 1 - Get Codebase to local workspace ()

[Terminal-Version]
   1. Open a `terminal` or `command prompt` 
   2. Run `git clone https://github.com/vincycode7/object_tracking_yolov5_deepsort.git`
   3. Once download is complete, change directory into the current working directory using `cd object_tracking_yolov5_deepsort/` as this directory contains all the code you need.

[Zip-version]
    1. Navigate to the `<> Code` Dropdown
    2. Click to view options available
    3. Click on the download dropdown to download a zipped file of all the code.
    4. Once download is complete, Unzip into a folder and open the parent folder `cd object_tracking_yolov5_deepsort` with an `IDE` of your choice.


## Step 2 - Basic OS Installation

[macOS]
   1. Install pip - `python -m ensurepip --upgrade`
   2. Install pipenv `pip3 install pipenv`
   3. Install xcode command line tool - `xcode-select --install`

[Ubuntu]
  1. Install pip - `sudo apt-get install python3-pip`
  2. Install pipenv - `pip3 install pipenv` or `sudo apt install pipenv`

## Step 3 - Create and activate an Environment for project

  1. Navigate to your project folder `cd object_tracking_yolov5_deepsort`
  2. Create a new Pipenv environment in that folder and activate that environment - `pipenv shell`

## Step 4 - Project Dependencies Installations
  1. install dependencies using   `pipenv install` or `pipenv run pip3 install -r requirements.txt`

## Step 5 - Running Project
  1. Confirm your current working directory is `object_tracking_yolov5_deepsort`
  2. Activate the environment using the command  `pipenv shell`
  3. Run the app using the command `streamlit run Home.py` or `reset && streamlit run Home.py`

Note: please upon first run, project will download model weight which would then be saved in this path `./model_wight`, also note that this download process might take few seconds.

Once project started running you will have a web interface pop-up, In case you do not see a pop-up check you terminal to see the  localhost and port in this format    
`Local URL: http://localhost:port or Network URL: http://host:port` copy any one of them and paste in your browser.

## Step 6 - Interacting with the User Interface(UI).
  1. On the UI you will see difference configuration options on the left panel to help set thresholds and load input data, do take a look at all available options before going to item `2` below.
  2. Once you have viewed all available options, Toggle the `Run Solution` checkbox to either run or stop running the solution.
  3. While program is running, you should see bounding box information displayed on the image, also tracking information are displayed below the image.

## Step 7 - Use case section and answer.
  1. Once you have succesfully ran the above section, You should also checkout the `Use case questoin` section on the UI, It answer various, simple questions about the system implementation and edge cases.

## Reference
1) [object_tracking_yolov5_deepsort] (https://github.com/vincycode7/object_tracking_yolov5_deepsort)
2) [Yolov5_DeepSort_Pytorch](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch)   
3) [yolov5](https://github.com/ultralytics/yolov5)  
4) [deep_sort_pytorch](https://github.com/ZQPei/deep_sort_pytorch)       
5) [deep_sort](https://github.com/nwojke/deep_sort)   

Note: please follow the [LICENCE](https://github.com/ultralytics/yolov5/blob/master/LICENSE) of YOLOv5! 

Test Video by <a href="https://pixabay.com/users/mikes-photography-1860391/?utm_source=link-attribution&amp;utm_medium=referral&amp;utm_campaign=video&amp;utm_content=2165">Mikes-Photography</a> from <a href="https://pixabay.com//?utm_source=link-attribution&amp;utm_medium=referral&amp;utm_campaign=video&amp;utm_content=2165">Pixabay</a>

# Thank you !!!
