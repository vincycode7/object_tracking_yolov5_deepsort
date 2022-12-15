import cv2, os, random
import sys
import torch
import json
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from detector_utils.general import (xyxy2xywh, xywh2xyxy)
from detector_utils.parser import get_config
from deep_sort import build_tracker
import os
import warnings
import cv2
import torch
import torch.backends.cudnn as cudnn

# Deep sort algorithm
class YoloBaseDetector(object):
    def __init__(self, **kwargs):
        """
            This is the base Yolov5 arch for this project
            It defines how data should flow in and out of
            the yolo model to suit app use case.
        """
        self.kwargs = kwargs
        self.img_size = kwargs.get('img_size', 640)
        self.device = kwargs.get('device', 'cpu') # 'cuda device, i.e. 0 or 0,1,2,3 or cpu'
        self.use_cuda = self.device == 'cpu' and torch.cuda.is_available()
        self.half = self.device != 'cpu'  # half precision only supported on CUDA
        self.load_class(**kwargs)
        self.load_model(**kwargs)

    def detect_objects(self,im0, classes=None, conf_thres=0.40, iou_thres=0.8):
        """
        :param im0: original image, BGR format
        :return:
        """
        self.detector.conf = conf_thres # NMS confidence threshold
        self.detector.iou = iou_thres
        self.detector.classes = classes
        self.detector.augment = self.kwargs.get('augment', True)

        with torch.no_grad():
            pred = self.detector(im0).pandas().xyxy[0].sort_values('xmin')  # sorted left-right  # list: bz * [ (#obj, 6)]

        if pred.shape[0] != 0:
            bbox_xywh = xyxy2xywh(pred[['xmin','ymin','xmax','ymax']].to_numpy()[:,:4])
            confs = pred[['confidence']].to_numpy()
            classes = pred[['class']].to_numpy()
        else:
            bbox_xywh = None
            confs = None
            classes = None

        return bbox_xywh, confs, classes

    def load_class(self,**kwargs):
        static_file_path= kwargs.get('static_file_path','static_files/yolo_classes.json')
        try:
            with open(static_file_path, 'r') as fp:
                self.key_to_string = json.load(fp)
            self.string_to_key = {value:key for key, value in self.key_to_string.items()}
            return (self.key_to_string, self.string_to_key)
        except Exception as e:
            raise Exception(f"Error {e} while loading model class")

    def load_model(self,**kwargs):
        use_cuda = self.device != 'cpu' and torch.cuda.is_available()

        # ***************************** initialize YOLO-V5 **********************************
        # self.detector = torch.load(kwargs.get('weights','yolov5/weights/yolov5s.pt'), map_location=self.device)['model'].float()  # load to FP32
        model_size = kwargs.get('model_size','yolov5n') #['yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']
        model_weight_path = kwargs.get('model_weight_path', "./model_weight/"+model_size+"_v2.pt") #'yolov5/weights/yolov5s.pt') "./model_weight/yolov5s.pt"
        pretrained_model = False if os.path.exists(model_weight_path) else True
        
        try:
            import sys
            sys.path.insert(0, './yolov5')
            self.detector = torch.hub.load('ultralytics/yolov5',model_size, pretrained=pretrained_model, force_reload=True, _verbose=pretrained_model) 
            # if self.pretrained_model else torch.hub.load('ultralytics/yolov5', 'yolov5s', path=self.url_to_model, force_reload=True)
            torch.save(self.detector, model_weight_path) if pretrained_model else None
            self.detector = torch.load(model_weight_path) if not pretrained_model else self.detector

            print("Model has been loaded")
            if os.path.isfile(model_size+'.pt'):
                os.remove(model_size+'.pt')
        except Exception as e:
            raise Exception(f"Exception {e} occured while trying to load model")

        if self.device == 'cpu':
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)
        return self

    def compute_color_for_labels(self,label):
        """
        Simple function that adds fixed color depending on the class
        """
        palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
        return tuple(color)
        
    def draw_boxes(self,img, bbox, identities=None, classes=None, offset=(0,0)):
        for i,box in enumerate(bbox):
            x1,y1,x2,y2 = [int(i) for i in box]
            x1 += offset[0]
            x2 += offset[0]
            y1 += offset[1]
            y2 += offset[1]
            # box text and bar
            id = int(identities[i]) if identities is not None else 0   
            classname = classes[i] if classes is not None else ""
            color = self.compute_color_for_labels(id)
            label = '{}{:d}'.format(classname+" ", id)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
            cv2.rectangle(img,(x1, y1),(x2,y2),color,3)
            cv2.rectangle(img,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)
            cv2.putText(img,label,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 2, [255,255,255], 2)

        return img
    
class YoloObjectTrackerFrame(YoloBaseDetector):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        """
            This is the object tracker, that is responsible for detecting and tracking objects
            to suit application needs.
        """
        # ***************************** initialize DeepSORT configs **********************************
        self.cfg = get_config()
        self.cfg_path = kwargs.get('config_deepsort', "./configs/deep_sort.yaml")
        # print(f"cfg_path: {self.cfg_path}")
        self.cfg.merge_from_file(config_file=self.cfg_path) 
        

    def _init_tracker(self, max_cosine_distance = 0.7,nn_budget = None, model_filename = 'model_data/mars-small128.pb',save_enc_img_feature=False):
        #initialize deep sort object
        deepsort = build_tracker(self.cfg, use_cuda=self.use_cuda,save_enc_img_feature=save_enc_img_feature)
        deepsort.sequence = 0
        deepsort.results = {'class_metric':{}, 'box_metric':{}}
        return deepsort

    def image_dectection(self,im0, classes=None, conf_thres=0.40, draw_box_on_img=True, iou_thres=0.8, deepsort_memory=None):
        """
        :param im0: original image, BGR format
        :return:
        """
        # Detection time *********************************************************
        # Inference
        bbox_xywh, confs,classes = self.detect_objects(im0=im0, classes=classes, conf_thres=conf_thres, iou_thres=iou_thres)
     
        if not isinstance(bbox_xywh, type(None)):
            # ****************************** deepsort ****************************
            # print("/n/n/n/nIn sort")
            loop_times = 3 if deepsort_memory.sequence <= 1 else 1
            for _ in range(loop_times):
                outputs = deepsort_memory.update(bbox_xywh[:, :4], confs, im0)
                deepsort_memory.sequence = deepsort_memory.sequence+ 1
            outputs = torch.Tensor(outputs).float()
            # (#ID, 5) x1,y1,x2,y2,track_ID
        else:
            outputs = torch.zeros((0, 5)).float()

        # print(f"deepsort output : {outputs}")
        if outputs.shape[0] == 0 and not isinstance(bbox_xywh, type(None)):
            outputs = xywh2xyxy(bbox_xywh[:, :4])
            tracking_ids = list(range(bbox_xywh.shape[0]))
        else:
            tracking_ids = list(outputs[:,-1])

        if not isinstance(type(outputs), type(None)) and outputs.shape[0] != 0:
        #     #assign unique id to unique classes
        #     # loop through all detectoin
        #     # get their class
        #     # check if their id is in the class id
        #     # if not increment class
        #     # add unique id to class
        #     # assign new unique id to elemete for printing
            class_names = []
            object_tracker_ids = []
            for each_idx in range(min(classes.shape[0], outputs.shape[0])):
        #         # print(self.key_to_string)
                class_name = self.key_to_string.get(str(int(classes[each_idx])))
                tracker_id = tracking_ids[each_idx]
                class_metric = deepsort_memory.results["class_metric"].get(class_name, {})
                box_metric_key_value = deepsort_memory.results.get("box_metric_key_value",[])

        #         # create class metric instance for current object if not available
                if class_metric == {}:
                    deepsort_memory.results["class_metric"][class_name] = {'class_count': 0, 'location_unique_id' : {}}           

        #         # check if current object id is in class detected key
                if int(tracker_id) not in deepsort_memory.results["class_metric"][class_name].get('location_unique_id',{}).keys():
                    if deepsort_memory.results["class_metric"][class_name].get("location_unique_id",{})=={}:
                        deepsort_memory.results["class_metric"][class_name]["location_unique_id"] = {}

                    deepsort_memory.results["class_metric"][class_name]["location_unique_id"][int(tracker_id)] = deepsort_memory.results["class_metric"][class_name]['class_count']
        #             # deepsort_memory.results["class_metric"][class_name]["class_value_key"][deepsort_memory.results["class_metric"][class_name]['class_count']] = tracker_id
                    deepsort_memory.results["class_metric"][class_name]['class_count'] += 1
                
                # outputs[each_idx,-1] = deepsort_memory.results["class_metric"][class_name]["location_unique_id"][int(tracker_id)]
                object_tracker_ids.append(deepsort_memory.results["class_metric"][class_name]["location_unique_id"][int(tracker_id)])
                class_names.append(class_name)
            minby = min(classes.shape[0], outputs.shape[0])
            im0 = self.draw_boxes(im0, outputs[:minby, :4], object_tracker_ids[:minby], classes=class_names)  # BGR
        # return im0,deepsort_memory, detection_time, tracking_time
        return im0, deepsort_memory






