

## Basic OS Installation
[macOS]
1. Install pip - `python -m ensurepip --upgrade`
2. Install pipenv `pip3 install pipenv`
3. Install xcode command line tool - `xcode-select --install`

[Ubuntu]
1. Install pip - `sudo apt-get install python3-pip`
2. Install pipenv - `pip3 install pipenv` or `sudo apt install pipenv`

## Create and activate an Environment for project
1. Navigate to your project folder `cd object_tracking_yolov5_deepsort`
2. Create a new Pipenv environment in that folder and activate that environment - `pipenv shell`

## Project Dependencies Installations
1. install dependencies using   `pipenv install` or `pipenv run pip3 install -r requirements.txt`

## Running Project
1. Confirm your current working directory is `object_tracking_yolov5`
2. Activate the environment using the command  `pipenv shell`
3. Run the app using the command `streamlit run Home.py` or `reset && streamlit run Home.py`

Note: please upon first run, project will download model weight which would then be saved in this path `./model_wight`, also note that this download process might take few seconds.

4. Once project started running you will have a web interface pop-up, In case you do not see a pop-up check you terminal to see the  localhost and port in this format    
`Local URL: http://localhost:port or Network URL: http://host:port` copy any one of them and paste in your browser.

## Reference
1) [Yolov5_DeepSort_Pytorch](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch)   
2) [yolov5](https://github.com/ultralytics/yolov5)  
3) [deep_sort_pytorch](https://github.com/ZQPei/deep_sort_pytorch)       
4) [deep_sort](https://github.com/nwojke/deep_sort)   

Note: please follow the [LICENCE](https://github.com/ultralytics/yolov5/blob/master/LICENSE) of YOLOv5! 

Test Video by <a href="https://pixabay.com/users/mikes-photography-1860391/?utm_source=link-attribution&amp;utm_medium=referral&amp;utm_campaign=video&amp;utm_content=2165">Mikes-Photography</a> from <a href="https://pixabay.com//?utm_source=link-attribution&amp;utm_medium=referral&amp;utm_campaign=video&amp;utm_content=2165">Pixabay</a>