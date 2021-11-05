###### EXTRACT NETWORKX GRAPHS BASED ON FLOORPLAN IMAGES
# -*- coding: utf-8 -*-
"""
Created on Feb 19 2021

@author: CHEN JIELIN
"""

import os, json, random, time
from skimage import transform
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import PIL
from PIL import Image
from collections import Counter

import time
from multiprocessing import cpu_count
from multiprocessing.dummy import Pool as ThreadPool

import torch
from torch.nn import functional as F
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import save_image
from torchvision.transforms import ColorJitter, ToTensor, RandomResizedCrop, Compose, Normalize, transforms, Grayscale, RandomGrayscale
import torch.nn.functional as F
from torch.utils.data import DataLoader
from mpl_toolkits.axes_grid1 import AxesGrid

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer

from Text_Segmentation.Dataloader import EvaluateSet
from Text_Segmentation.models.text_segmentation import TextSegament, XceptionTextSegment

from floortrans.models import get_model
from floortrans.loaders import FloorplanSVG, DictToTensor, Compose, RotateNTurns
from floortrans.plotting import segmentation_plot, polygons_to_image, draw_junction_from_dict, discrete_cmap
discrete_cmap()
from floortrans.post_prosessing import split_prediction, get_polygons, split_validation

import geopandas as gpd
import rasterio
from rasterio import features
import fiona
from affine import Affine
import networkx as nx
import shapely
from shapely.geometry import Polygon as sPolygon
from shapely.geometry import shape
from shapely.geometry import LineString as sLine
from shapely.geometry import Point as sPoint
from shapely.geometry import MultiPoint, MultiPolygon,MultiLineString
from shapely.strtree import STRtree

from utils import *
from methods import *


start_time = time.time()

##### Load models
rot = RotateNTurns()
room_classes = ["Background", "Outdoor", "Wall", "Kitchen", "Living Room" ,"Bed Room", "Bath", "Entry", "Railing", "Storage", "Garage", "Undefined"]
icon_classes = ["No Icon", "Window", "Door", "Closet", "Electrical Applience" ,"Toilet", "Sink", "Sauna Bench", "Fire Place", "Bathtub", "Chimney"]
CubiCasa5k_model = get_model('hg_furukawa_original', 51)
n_classes = 44
n_rooms = 12
n_icons = 11
split = [21, 12, 11]
CubiCasa5k_model.conv4_ = torch.nn.Conv2d(256, n_classes, bias=True, kernel_size=1)
CubiCasa5k_model.upsample = torch.nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=4)
checkpoint = torch.load('./ckpt/CubiCasa5k/model_best_val_loss_var.pkl')
CubiCasa5k_model.load_state_dict(checkpoint['model_state'])
CubiCasa5k_model.eval()
CubiCasa5k_model.cuda()
print("CubiCasa5k Model for walls/doors/windows/rooms loaded.")


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
cfg.MODEL.WEIGHTS = os.path.join("./ckpt/CVCFP_wdw/1", "model_0009999.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
wdw_predictor_1 = DefaultPredictor(cfg)

cfg.MODEL.WEIGHTS = os.path.join("./ckpt/CVCFP_wdw/2", "model_0009999.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
wdw_predictor_2 = DefaultPredictor(cfg)

cfg.MODEL.WEIGHTS = os.path.join("./ckpt/CVCFP_wdw/3", "model_0009999.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
wdw_predictor_3 = DefaultPredictor(cfg)

cfg.MODEL.WEIGHTS = os.path.join("./ckpt/CVCFP_wdw/4", "model_0009999.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
wdw_predictor_4 = DefaultPredictor(cfg)

cfg.MODEL.WEIGHTS = os.path.join("./ckpt/CVCFP_wdw/5", "model_0009999.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
wdw_predictor_5 = DefaultPredictor(cfg)
print("Detectron 2 Mask R-CNN for Wall/Door/Windows loaded.")


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.WEIGHTS = os.path.join("./ckpt/CVCFP_stairs/1", "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
stairs_predictor_1 = DefaultPredictor(cfg)

cfg.MODEL.WEIGHTS = os.path.join("./ckpt/CVCFP_stairs/2", "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
stairs_predictor_2 = DefaultPredictor(cfg)

cfg.MODEL.WEIGHTS = os.path.join("./ckpt/CVCFP_stairs/3", "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
stairs_predictor_3 = DefaultPredictor(cfg)

cfg.MODEL.WEIGHTS = os.path.join("./ckpt/CVCFP_stairs/4", "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
stairs_predictor_4 = DefaultPredictor(cfg)

cfg.MODEL.WEIGHTS = os.path.join("./ckpt/CVCFP_stairs/5", "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
stairs_predictor_5 = DefaultPredictor(cfg)
print("Detectron 2 Mask R-CNN for Stairs loaded.")
print("LOAD MODELS--- %s seconds ---" % (time.time() - start_time))

## LOAD IMAGES ##

def graph_extractor(img_dir, output_path):
    img = Image.open(img_dir)
    im = img.convert('RGB')
    im = np.array(im)
    h,w=im.shape[:2]

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Detect the background of image, if black, reverse
    # Binarize the image
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    _, im_b = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)#binarization

    if dict(Counter(list(np.array(im_b).flat)))[255] > dict(Counter(list(np.array(im_b).flat)))[0]:
        pass
    if dict(Counter(list(np.array(im_b).flat)))[255] < dict(Counter(list(np.array(im_b).flat)))[0]:
        im = 255 - im
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        _, im_b = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)#binarization

    print("Detect the background/Binarization--- %s seconds ---" % (time.time() - start_time))


    # Filter out text
    masked_im = im_b.copy()
    prob_thre = 0.5
    model_txt_1 = XceptionTextSegment()
    model_txt_1.load_state_dict(torch.load("./ckpt/text/text_seg_model_590epos.pt", map_location='cpu'))
    model_txt_1 = model_txt_1.cuda()
    model_txt_2 = XceptionTextSegment()
    model_txt_2.load_state_dict(torch.load("./ckpt/text/text_seg_model_681epos.pt", map_location='cpu'))
    model_txt_2 = model_txt_2.cuda()
    rs = np.arange(0, 180, 45)
    for r in rs:
        im_ = rotate(im,r)
        im_txt = Image.fromarray(im_)
        im_txt.save('./Text_Segmentation/test_data/' + 'im_text.jpg')
        evalset = EvaluateSet(mean=[0.4935, 0.4563, 0.4544],
                            std=[0.3769, 0.3615, 0.3566],
                            img_folder='./Text_Segmentation/test_data',
                            resize=600)
        #model_1
        for i in evalset:
            (img, origin, unpadder), file_name = i
            with torch.no_grad():
                out = model_txt_1(img.to('cuda'))
            prob = F.sigmoid(out)
            mask = prob > prob_thre
            mask = torch.nn.MaxPool2d(kernel_size=(3, 3), padding=(1, 1), stride=1)(mask.float()).byte()
            mask = unpadder(mask).float().cpu().numpy().squeeze().transpose(1,2,0)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)#binarization
            mask = rotate(mask,-r)
        masked_im[mask==1] = 255
        #model_2
        for i in evalset:
            (img, origin, unpadder), file_name = i
            with torch.no_grad():
                out = model_txt_2(img.to('cuda'))
            prob = F.sigmoid(out)
            mask = prob > prob_thre
            mask = torch.nn.MaxPool2d(kernel_size=(3, 3), padding=(1, 1), stride=1)(mask.float()).byte()
            mask = unpadder(mask).float().cpu().numpy().squeeze().transpose(1,2,0)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)#binarization
            mask = rotate(mask,-r)
        masked_im[mask==1] = 255

    masked_im_gray = np.zeros((h,w,3), np.uint8)
    masked_im_gray[:,:,0]=masked_im
    masked_im_gray[:,:,1]=masked_im
    masked_im_gray[:,:,2]=masked_im

    print("Filter out text--- %s seconds ---" % (time.time() - start_time))

    # Extract mask background
    k_size = 20 #3
    kernel = np.ones((k_size,k_size), np.uint8)
    masked_im_ = cv2.morphologyEx(masked_im, cv2.MORPH_OPEN, kernel,iterations = 1)
    mask_background, mask_area = vectorization_mask(masked_im_)
    FP_area = h*w - mask_area
    k_size = 20 #3
    kernel = np.ones((k_size,k_size), np.uint8)
    mask_background_ = cv2.morphologyEx(mask_background, cv2.MORPH_CLOSE, kernel,iterations = 1)

    print("Extract mask background--- %s seconds ---" % (time.time() - start_time))

    ### Extract floorplan information ###

    # CubiCasa5k_model
    im_torch_original = np.moveaxis(im, -1, 0) # correct the dimension order
    im_torch_filtered = np.moveaxis(masked_im_gray, -1, 0) # use the clean fp
    im_norm_original = 2 * (im_torch_original / 255.0) - 1  # Normalization values to range -1 and 1
    im_norm_original = torch.from_numpy(im_norm_original).unsqueeze(0)
    im_norm_filtered = 2 * (im_torch_filtered / 255.0) - 1  # Normalization values to range -1 and 1
    im_norm_filtered = torch.from_numpy(im_norm_filtered).unsqueeze(0)

    with torch.no_grad():
        im_norm = im_norm_filtered.type('torch.FloatTensor').cuda()
        _, _, height, width = im_norm.shape
        img_size = (height,width)
        rotations = [(0, 0), (1, -1), (2, 2), (-1, 1)]
        pred_count = len(rotations)
        prediction = torch.zeros([pred_count, n_classes, height, width])
        for i, r in enumerate(rotations):
            forward, back = r
            # We rotate first the image
            rot_image = rot(im_norm, 'tensor', forward)
            pred = CubiCasa5k_model(rot_image)

            # We rotate prediction back
            pred = rot(pred, 'tensor', back)
            # We fix heatmaps
            pred = rot(pred, 'points', back)
            # We make sure the size is correct
            pred = F.interpolate(pred, size=img_size, mode='bilinear', align_corners=True)
            # We add the prediction to output
            prediction[i] = pred[0]

    prediction = torch.mean(prediction, 0, True)
    rooms_pred = F.softmax(prediction[0, 21:21+12], 0).cpu().data.numpy()
    rooms_pred = np.argmax(rooms_pred, axis=0)

    with torch.no_grad():
        im_norm = im_norm_original.type('torch.FloatTensor').cuda()
        _, _, height, width = im_norm.shape
        img_size = (height,width)
        
        rotations = [(0, 0), (1, -1), (2, 2), (-1, 1)]
        pred_count = len(rotations)
        prediction = torch.zeros([pred_count, n_classes, height, width])
        for i, r in enumerate(rotations):
            forward, back = r
            # We rotate first the image
            rot_image = rot(im_norm, 'tensor', forward)
            pred = CubiCasa5k_model(rot_image)

            # We rotate prediction back
            pred = rot(pred, 'tensor', back)
            # We fix heatmaps
            pred = rot(pred, 'points', back)
            # We make sure the size is correct
            pred = F.interpolate(pred, size=img_size, mode='bilinear', align_corners=True)
            # We add the prediction to output
            prediction[i] = pred[0]

    prediction = torch.mean(prediction, 0, True)
    icons_pred = F.softmax(prediction[0, 21+12:], 0).cpu().data.numpy()
    icons_pred = np.argmax(icons_pred, axis=0)

    walls = (rooms_pred == 2).astype(int)
    walls = (walls != 0).astype(int)
    k_size = 5 #3
    kernel = np.ones((k_size,k_size), np.uint8)
    walls_ = cv2.dilate(walls.astype('uint8'), kernel,iterations = 1)

    doors = (icons_pred == 2).astype(int)
    windows = (icons_pred == 1).astype(int)
    k_size = 25 #40
    kernel = np.ones((k_size,k_size), np.uint8)
    doors_ = cv2.dilate(doors.astype('uint8'), kernel,iterations = 1)
    windows_ = cv2.dilate(windows.astype('uint8'), kernel,iterations = 1)

    print("CubiCasa5k_model--- %s seconds ---" % (time.time() - start_time))

    # Mask R-CNN: Wall/Door/Window
    score_thre_wall = 0.70
    score_thre_door_window = 0.90
    r_list = np.arange(0, 180, 15)

    wall_instances = []
    door_instances = []
    window_instances = []

    try:
        for r in r_list:
            im_ = rotate(im,r)
            outputs_1 = wdw_predictor_1(im_)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
            wdw_1 = outputs_1['instances'].pred_masks.cpu().numpy().astype(int).squeeze()
            wdw_class_1 = outputs_1['instances'].pred_classes.cpu().numpy().astype(int).squeeze()
            wdw_score_1 = outputs_1['instances'].scores.cpu().numpy()
            for i in range(len(wdw_class_1)):
                if wdw_class_1[i] == 0: #or wdw_class_1[i] == 3:
                    if wdw_score_1[i] > score_thre_wall:
                        wall_instances.append(rotate(wdw_1[i].astype('uint8'),-r))
                if wdw_class_1[i] == 1 or wdw_class_1[i] == 3:
                    if wdw_score_1[i] > score_thre_door_window:
                        door_instances.append(rotate(wdw_1[i].astype('uint8'),-r))
                if wdw_class_1[i] == 2:
                    if wdw_score_1[i] > score_thre_door_window:
                        window_instances.append(rotate(wdw_1[i].astype('uint8'),-r))
        for r in r_list:
            im_ = rotate(masked_im_gray,r)
            outputs_1 = wdw_predictor_1(im_)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
            wdw_1 = outputs_1['instances'].pred_masks.cpu().numpy().astype(int).squeeze()
            wdw_class_1 = outputs_1['instances'].pred_classes.cpu().numpy().astype(int).squeeze()
            wdw_score_1 = outputs_1['instances'].scores.cpu().numpy()
            for i in range(len(wdw_class_1)):
                if wdw_class_1[i] == 0: # or wdw_class_1[i] == 3:
                    if wdw_score_1[i] > score_thre_wall:
                        wall_instances.append(rotate(wdw_1[i].astype('uint8'),-r))
                if wdw_class_1[i] == 1 or wdw_class_1[i] == 3:
                    if wdw_score_1[i] > score_thre_door_window:
                        door_instances.append(rotate(wdw_1[i].astype('uint8'),-r))
                if wdw_class_1[i] == 2:
                    if wdw_score_1[i] > score_thre_door_window:
                        window_instances.append(rotate(wdw_1[i].astype('uint8'),-r))
    except:
        pass

    try:
        for r in r_list:
            im_ = rotate(im,r)
            outputs_2 = wdw_predictor_2(im_)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
            wdw_2 = outputs_2['instances'].pred_masks.cpu().numpy().astype(int).squeeze()
            wdw_class_2 = outputs_2['instances'].pred_classes.cpu().numpy().astype(int).squeeze()
            wdw_score_2 = outputs_2['instances'].scores.cpu().numpy()
            for i in range(len(wdw_class_2)):
                if wdw_class_2[i] == 0: # or wdw_class_2[i] == 3:
                    if wdw_score_2[i] > score_thre_wall:
                        wall_instances.append(rotate(wdw_2[i].astype('uint8'),-r))
                if wdw_class_2[i] == 1 or wdw_class_1[i] == 3:
                    if wdw_score_2[i] > score_thre_door_window:
                        door_instances.append(rotate(wdw_2[i].astype('uint8'),-r))
                if wdw_class_2[i] == 2:
                    if wdw_score_2[i] > score_thre_door_window:
                        window_instances.append(rotate(wdw_2[i].astype('uint8'),-r))
                        
        for r in r_list:
            im_ = rotate(masked_im_gray,r)
            outputs_2 = wdw_predictor_2(im_)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
            wdw_2 = outputs_2['instances'].pred_masks.cpu().numpy().astype(int).squeeze()
            wdw_class_2 = outputs_2['instances'].pred_classes.cpu().numpy().astype(int).squeeze()
            wdw_score_2 = outputs_2['instances'].scores.cpu().numpy()
            for i in range(len(wdw_class_2)):
                if wdw_class_2[i] == 0: # or wdw_class_2[i] == 3:
                    if wdw_score_2[i] > score_thre_wall:
                        wall_instances.append(rotate(wdw_2[i].astype('uint8'),-r))
                if wdw_class_2[i] == 1 or wdw_class_1[i] == 3:
                    if wdw_score_2[i] > score_thre_door_window:
                        door_instances.append(rotate(wdw_2[i].astype('uint8'),-r))
                if wdw_class_2[i] == 2:
                    if wdw_score_2[i] > score_thre_door_window:
                        window_instances.append(rotate(wdw_2[i].astype('uint8'),-r))
    except:
        pass

    try:
        for r in r_list:
            im_ = rotate(im,r)
            outputs_3 = wdw_predictor_3(im_)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
            wdw_3 = outputs_3['instances'].pred_masks.cpu().numpy().astype(int).squeeze()
            wdw_class_3 = outputs_3['instances'].pred_classes.cpu().numpy().astype(int).squeeze()
            wdw_score_3 = outputs_3['instances'].scores.cpu().numpy()
            for i in range(len(wdw_class_3)):
                if wdw_class_3[i] == 0: # or wdw_class_3[i] == 3:
                    if wdw_score_3[i] > score_thre_wall:
                        wall_instances.append(rotate(wdw_3[i].astype('uint8'),-r))
                if wdw_class_3[i] == 1 or wdw_class_1[i] == 3:
                    if wdw_score_3[i] > score_thre_door_window:
                        door_instances.append(rotate(wdw_3[i].astype('uint8'),-r))
                if wdw_class_3[i] == 2:
                    if wdw_score_3[i] > score_thre_door_window:
                        window_instances.append(rotate(wdw_3[i].astype('uint8'),-r))
                        
        for r in r_list:
            im_ = rotate(masked_im_gray,r)
            outputs_3 = wdw_predictor_3(im_)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
            wdw_3 = outputs_3['instances'].pred_masks.cpu().numpy().astype(int).squeeze()
            wdw_class_3 = outputs_3['instances'].pred_classes.cpu().numpy().astype(int).squeeze()
            wdw_score_3 = outputs_3['instances'].scores.cpu().numpy()
            for i in range(len(wdw_class_3)):
                if wdw_class_3[i] == 0: # or wdw_class_3[i] == 3:
                    if wdw_score_3[i] > score_thre_wall:
                        wall_instances.append(rotate(wdw_3[i].astype('uint8'),-r))
                if wdw_class_3[i] == 1 or wdw_class_1[i] == 3:
                    if wdw_score_3[i] > score_thre_door_window:
                        door_instances.append(rotate(wdw_3[i].astype('uint8'),-r))
                if wdw_class_3[i] == 2:
                    if wdw_score_3[i] > score_thre_door_window:
                        window_instances.append(rotate(wdw_3[i].astype('uint8'),-r))
    except:
        pass

    try:
        for r in r_list:
            im_ = rotate(im,r)
            outputs_4 = wdw_predictor_4(im_)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
            wdw_4 = outputs_4['instances'].pred_masks.cpu().numpy().astype(int).squeeze()
            wdw_class_4 = outputs_4['instances'].pred_classes.cpu().numpy().astype(int).squeeze()
            wdw_score_4 = outputs_4['instances'].scores.cpu().numpy()
            for i in range(len(wdw_class_4)):
                if wdw_class_4[i] == 0: # or wdw_class_4[i] == 3:
                    if wdw_score_4[i] > score_thre_wall:
                        wall_instances.append(rotate(wdw_4[i].astype('uint8'),-r))
                if wdw_class_4[i] == 1 or wdw_class_1[i] == 3:
                    if wdw_score_4[i] > score_thre_door_window:
                        door_instances.append(rotate(wdw_4[i].astype('uint8'),-r))
                if wdw_class_4[i] == 2:
                    if wdw_score_4[i] > score_thre_door_window:
                        window_instances.append(rotate(wdw_4[i].astype('uint8'),-r))
                        
        for r in r_list:
            im_ = rotate(masked_im_gray,r)
            outputs_4 = wdw_predictor_4(im_)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
            wdw_4 = outputs_4['instances'].pred_masks.cpu().numpy().astype(int).squeeze()
            wdw_class_4 = outputs_4['instances'].pred_classes.cpu().numpy().astype(int).squeeze()
            wdw_score_4 = outputs_4['instances'].scores.cpu().numpy()
            for i in range(len(wdw_class_4)):
                if wdw_class_4[i] == 0: # or wdw_class_4[i] == 3:
                    if wdw_score_4[i] > score_thre_wall:
                        wall_instances.append(rotate(wdw_4[i].astype('uint8'),-r))
                if wdw_class_4[i] == 1 or wdw_class_1[i] == 3:
                    if wdw_score_4[i] > score_thre_door_window:
                        door_instances.append(rotate(wdw_4[i].astype('uint8'),-r))
                if wdw_class_4[i] == 2:
                    if wdw_score_4[i] > score_thre_door_window:
                        window_instances.append(rotate(wdw_4[i].astype('uint8'),-r))
    except:
        pass

    try:
        for r in r_list:
            im_ = rotate(im,r)
            outputs_5 = wdw_predictor_5(im_)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
            wdw_5 = outputs_5['instances'].pred_masks.cpu().numpy().astype(int).squeeze()
            wdw_class_5 = outputs_5['instances'].pred_classes.cpu().numpy().astype(int).squeeze()
            wdw_score_5 = outputs_5['instances'].scores.cpu().numpy()
            for i in range(len(wdw_class_5)):
                if wdw_class_5[i] == 0: # or wdw_class_5[i] == 3:
                    if wdw_score_5[i] > score_thre_wall:
                        wall_instances.append(rotate(wdw_5[i].astype('uint8'),-r))
                if wdw_class_5[i] == 1 or wdw_class_1[i] == 3:
                    if wdw_score_5[i] > score_thre_door_window:
                        door_instances.append(rotate(wdw_5[i].astype('uint8'),-r))
                if wdw_class_5[i] == 2:
                    if wdw_score_5[i] > score_thre_door_window:
                        window_instances.append(rotate(wdw_5[i].astype('uint8'),-r))
                        
        for r in r_list:
            im_ = rotate(masked_im_gray,r)
            outputs_5 = wdw_predictor_5(im_)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
            wdw_5 = outputs_5['instances'].pred_masks.cpu().numpy().astype(int).squeeze()
            wdw_class_5 = outputs_5['instances'].pred_classes.cpu().numpy().astype(int).squeeze()
            wdw_score_5 = outputs_5['instances'].scores.cpu().numpy()
            for i in range(len(wdw_class_5)):
                if wdw_class_5[i] == 0: # or wdw_class_5[i] == 3:
                    if wdw_score_5[i] > score_thre_wall:
                        wall_instances.append(rotate(wdw_5[i].astype('uint8'),-r))
                if wdw_class_5[i] == 1 or wdw_class_1[i] == 3:
                    if wdw_score_5[i] > score_thre_door_window:
                        door_instances.append(rotate(wdw_5[i].astype('uint8'),-r))
                if wdw_class_5[i] == 2:
                    if wdw_score_5[i] > score_thre_door_window:
                        window_instances.append(rotate(wdw_5[i].astype('uint8'),-r))
    except:
        pass

    wall_rcnn_sum = np.zeros(im.shape[:2]).astype('uint8')
    for i in wall_instances:
        wall_rcnn_sum = wall_rcnn_sum + i

    door_rcnn_sum = np.zeros(im.shape[:2]).astype('uint8')
    for i in door_instances:
        door_rcnn_sum = door_rcnn_sum + i

    window_rcnn_sum = np.zeros(im.shape[:2]).astype('uint8')
    for i in window_instances:
        window_rcnn_sum = window_rcnn_sum + i

    # wall_rcnn_sum = (wall_rcnn_sum != 0).astype(int)
    door_rcnn_sum = (door_rcnn_sum != 0).astype(int)
    window_rcnn_sum = (window_rcnn_sum != 0).astype(int)

    print("Mask R-CNN: Wall/Door/Window--- %s seconds ---" % (time.time() - start_time))


    # Mask R-CNN: Stairs
    outputs_1 = stairs_predictor_1(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    stairs_1 = outputs_1['instances'].pred_masks.cpu().numpy().astype(int).squeeze()
    scores_1 = outputs_1['instances'].scores.cpu().numpy()
    if 1 in stairs_1:
        if len(stairs_1.shape) == 2:
            pass
        elif len(stairs_1.shape) != 2:
            stairs_o = stairs_1
            stairs_1 = np.zeros(im.shape[:2])
            for i in range(len(scores_1)):
                if scores_1[i] > 0.90:
                    stairs_1 = stairs_1 + stairs_o[i]
    else:
        stairs_1 = np.zeros(im.shape[:2])
    stairs_1 = (stairs_1 != 0).astype(int)

    outputs_2 = stairs_predictor_2(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    stairs_2 = outputs_2['instances'].pred_masks.cpu().numpy().astype(int).squeeze()
    scores_2 = outputs_2['instances'].scores.cpu().numpy()
    if 1 in stairs_2:
        if len(stairs_2.shape) == 2:
            pass
        elif len(stairs_2.shape) != 2:
            stairs_o = stairs_2
            stairs_2 = np.zeros(im.shape[:2])
            for i in range(len(scores_2)):
                if scores_2[i] > 0.90:
                    stairs_2 = stairs_2 + stairs_o[i]
    else:
        stairs_2 = np.zeros(im.shape[:2])
    stairs_2 = (stairs_2 != 0).astype(int)

    outputs_3 = stairs_predictor_3(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    stairs_3 = outputs_3['instances'].pred_masks.cpu().numpy().astype(int).squeeze()
    scores_3 = outputs_3['instances'].scores.cpu().numpy()
    if 1 in stairs_3:
        if len(stairs_3.shape) == 2:
            pass
        elif len(stairs_3.shape) != 2:
            stairs_o = stairs_3
            stairs_3 = np.zeros(im.shape[:2])
            for i in range(len(scores_3)):
                if scores_3[i] > 0.90:
                    stairs_3 = stairs_3 + stairs_o[i]
    else:
        stairs_3 = np.zeros(im.shape[:2])
    stairs_3 = (stairs_3 != 0).astype(int)

    outputs_4 = stairs_predictor_4(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    stairs_4 = outputs_4['instances'].pred_masks.cpu().numpy().astype(int).squeeze()
    scores_4 = outputs_4['instances'].scores.cpu().numpy()
    if 1 in stairs_4:
        if len(stairs_4.shape) == 2:
            pass
        elif len(stairs_4.shape) != 2:
            stairs_o = stairs_4
            stairs_4 = np.zeros(im.shape[:2])
            for i in range(len(scores_4)):
                if scores_4[i] > 0.90:
                    stairs_4 = stairs_4 + stairs_o[i]
    else:
        stairs_4 = np.zeros(im.shape[:2])
    stairs_4 = (stairs_4 != 0).astype(int)

    outputs_5 = stairs_predictor_5(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    stairs_5 = outputs_5['instances'].pred_masks.cpu().numpy().astype(int).squeeze()
    scores_5 = outputs_5['instances'].scores.cpu().numpy()
    if 1 in stairs_5:
        if len(stairs_5.shape) == 2:
            pass
        elif len(stairs_5.shape) != 2:
            stairs_o = stairs_5
            stairs_5 = np.zeros(im.shape[:2])
            for i in range(len(scores_5)):
                if scores_5[i] > 0.90:
                    stairs_5 = stairs_5 + stairs_o[i]
    else:
        stairs_5 = np.zeros(im.shape[:2])
    stairs_5 = (stairs_5 != 0).astype(int)

    print("Mask R-CNN: Stairs--- %s seconds ---" % (time.time() - start_time))


    ## Wall gaps finalization
    # Simplify and close gaps of walls detected by Mask R-CNN
    k_size = 15
    scale_1 = 1.03
    scale_2 = 1.0
    rectangles = wall_instances_simplification(wall_instances,h,w,k_size,scale_1,scale_2)
    aff = Affine(1, 0, 1, 0, -1, 0) # Adjust flip and symmetry
    wall_rcnn_sum_e = features.rasterize(
                (MultiPolygon(rectangles)), 
                all_touched=True, 
                out_shape=(h,w),
                transform = aff,
                )
    # Sum walls of Mask R-CNN and CubiCasa5k Model
    walls_sum = walls + wall_rcnn_sum_e
    walls_sum = (walls_sum != 0).astype(int)
    mask = walls_sum == 1
    shapes_walls = features.shapes(walls_sum.astype('uint8'), mask=mask, transform = aff)
    walls_sum_instances = []
    max_area = h*w
    area_min_ratio = 0.0001
    for s, v in shapes_walls:
        ## polygon without holes
        if np.array(s['coordinates']).shape[0] == 1:
            polygon = sPolygon(np.array(s['coordinates']).squeeze())
            if polygon.area > max_area * area_min_ratio:
                walls_sum_instances.append(polygon)
        ## polygon with holes
        if np.array(s['coordinates']).shape[0] != 1:
            exterior = np.array(np.array(s['coordinates'])[0]).squeeze()
            interior = [np.array(p).squeeze() for p in np.array(s['coordinates'])[1:]]
            polygon = sPolygon(exterior, interior)
            if polygon.area > max_area * area_min_ratio:
                walls_sum_instances.append(polygon)
    # smooth out walls
    crack_0 = 0.00005 # larger is smoother
    crack_1 = 0.001
    crack_2 = 0.006
    crack_3 = 0.02
    walls_intances_smoothed = []
    for wall in walls_sum_instances:
        # check the complexity of the wall geometry
        if wall.area > FP_area*0.01:
            crack = crack_0
        if wall.area > FP_area*0.005 and wall.area <= FP_area*0.01:
            crack = crack_1
        if wall.area > FP_area*0.001 and wall.area <= FP_area*0.005:
            crack = crack_2
        elif wall.area <= FP_area*0.001:
            crack = crack_3
        # smooth the wall geometry
        t = 0
        while t < 4:
            wall, is_simple = wall_smooth(wall,crack_triangle_ratio=crack)
            t += 1
        walls_intances_smoothed.append(wall)
    walls_final = features.rasterize(
                (walls_intances_smoothed),
                all_touched=True, 
                out_shape=(h,w),
                transform = aff,
                )

    k_size = 3
    kernel = np.ones((k_size,k_size), np.uint8)
    walls_final_s = cv2.dilate(walls_final, kernel,iterations = 2)
    walls_final_s = cv2.morphologyEx(walls_final_s, cv2.MORPH_CLOSE, kernel,iterations = 3)
    
    print("Wall gap finalization--- %s seconds ---" % (time.time() - start_time))


    ## floor plan elements integration
    # 1: wall
    # 2: room
    # 3: door
    # 4: window
    # 5: stairs    
    rooms_ = mask_background_ + walls_final_s
    rooms_ = (rooms_ == 0).astype(int)
    rooms_ = (rooms_ != 0).astype(int)
    rooms_ = rooms_ * 2

    walls_final_s =  (walls_final_s != 0).astype(int)
    walls_rooms = rooms_ + walls_final_s
    walls_rooms[walls_rooms==3] = 1

    k_size = 10
    kernel = np.ones((k_size,k_size), np.uint8)
    door_rcnn_sum_ = cv2.dilate(door_rcnn_sum.astype('uint8'), kernel,iterations = 1)
    window_rcnn_sum_ = cv2.dilate(window_rcnn_sum.astype('uint8'), kernel,iterations = 1)
    doors = (doors_ + door_rcnn_sum_ != 0).astype(int)
    windows = (windows_ + window_rcnn_sum_ != 0).astype(int)
    doors = doors * 3
    windows = windows * 4
    doors_windows = doors + windows
    doors_windows[doors_windows==7] = 3

    stairs = stairs_1 + stairs_2 + stairs_3 + stairs_4 + stairs_5
    stairs = (stairs == 5).astype(int)
    # stairs = (stairs != 0).astype(int)
    stairs = stairs * 5

    FP = walls_rooms - doors_windows - stairs
    FP[FP < 0] = 0
    FP = FP + doors_windows + stairs
    FP[FP > 5] = 5

    k_1 = 3
    k_2 = 10
    kernel_1 = np.ones((k_1,k_1), np.uint8)
    kernel_2 = np.ones((k_2,k_2), np.uint8)
    img_v = cv2.erode(FP.astype('uint8'),kernel_1,iterations = 1)
    img_v = cv2.morphologyEx(img_v, cv2.MORPH_CLOSE, kernel_2,iterations = 1)

    print("Fuse FP information--- %s seconds ---" % (time.time() - start_time))


    ####### Vectorization ########

    ### Regional adjacency graph
    vectorization(img_v, output_path, img_dir, min_area = 100, min_room = 5000)
    print("Vectorization--- %s seconds ---" % (time.time() - start_time))
    G, nodes, edges = build_FPgraph_RAG(output_path, output_path, img_dir)
    print("build_FPgraph_RAG--- %s seconds ---" % (time.time() - start_time))

    ### Simple graph
    doors_idx = []
    windows_idx = []
    walls_idx = []
    spaces_idx = []
    for i in G.nodes:
        if G.nodes[i]['class_id'] == 1:
            walls_idx.append(i)
        if G.nodes[i]['class_id'] == 3:
            doors_idx.append(i)
        if G.nodes[i]['class_id'] == 4:
            windows_idx.append(i)
        if G.nodes[i]['class_id'] == 0 or G.nodes[i]['class_id'] == 2 or G.nodes[i]['class_id'] == 5:
            spaces_idx.append(i)

    door_edges = []
    window_edges = []
    wall_edges = []
    direct_edges = []
    for i in G.edges:
        if i[0] in doors_idx or i[1] in doors_idx:
            door_edges.append(i)
        if i[0] in windows_idx or i[1] in windows_idx:
            window_edges.append(i)
        if i[0] in walls_idx or i[1] in walls_idx:
            wall_edges.append(i)
        if i[0] in spaces_idx and i[1] in spaces_idx:
            direct_edges.append(i)

    # remake the graph
    G_n = nx.Graph()
    for i in nodes:
        if nodes[i]['class_id'] == 2 or nodes[i]['class_id'] == 5 or nodes[i]['class_id'] == 0:
            G_n.add_node(i, polygon=nodes[i]['polygon'], point=nodes[i]['point'], area=nodes[i]['area'], class_name=nodes[i]['class_name'])

    # Calculate mean room distance
    try:
        room_distances = []
        for i in G.nodes:
            if G.nodes[i]['class_id'] == 2:
                for j in range(i+1, len(G.nodes)):
                    if G.nodes[j]['class_id'] == 2:
                        distance = sPoint(G.nodes[i]['point']).distance(sPoint(G.nodes[j]['point']))
                        room_distances.append(distance)
        dis_thre = np.percentile(np.array(room_distances),50)
    except:
        dis_thre = 0

    # link spaces with wall edges
    for i in range(len(wall_edges)):
        for j in range(i+1,len(wall_edges)):
            if wall_edges[i][0] == wall_edges[j][0] and wall_edges[i][0] in walls_idx and wall_edges[i][1] in spaces_idx and wall_edges[j][1] in spaces_idx:
                curtuple = (wall_edges[i][1], wall_edges[j][1])
                distance = sPoint(G_n.nodes[curtuple[0]]['point']).distance(sPoint(G_n.nodes[curtuple[1]]['point']))
                if distance < dis_thre:
                    if curtuple not in G_n.edges.values() and (curtuple[1], curtuple[0]) not in G_n.edges.values() and curtuple[0] != curtuple[1]:
                        G_n.add_edge(curtuple[0], curtuple[1], edge_class='wall')
            if wall_edges[i][0] == wall_edges[j][1] and wall_edges[i][0] in walls_idx and wall_edges[i][1] in spaces_idx and wall_edges[j][0] in spaces_idx:
                curtuple = (wall_edges[i][1], wall_edges[j][0])
                distance = sPoint(G_n.nodes[curtuple[0]]['point']).distance(sPoint(G_n.nodes[curtuple[1]]['point']))
                if distance < dis_thre:
                    if curtuple not in G_n.edges.values() and (curtuple[1], curtuple[0]) not in G_n.edges.values() and curtuple[0] != curtuple[1]:
                        G_n.add_edge(curtuple[0], curtuple[1], edge_class='wall')
            if wall_edges[i][1] == wall_edges[j][0] and wall_edges[i][1] in walls_idx and wall_edges[i][0] in spaces_idx and wall_edges[j][1] in spaces_idx:
                curtuple = (wall_edges[i][0], wall_edges[j][1])
                distance = sPoint(G_n.nodes[curtuple[0]]['point']).distance(sPoint(G_n.nodes[curtuple[1]]['point']))
                if distance < dis_thre:
                    if curtuple not in G_n.edges.values() and (curtuple[1], curtuple[0]) not in G_n.edges.values() and curtuple[0] != curtuple[1]:
                        G_n.add_edge(curtuple[0], curtuple[1], edge_class='wall')
            if wall_edges[i][1] == wall_edges[j][1] and wall_edges[i][1] in walls_idx and wall_edges[i][0] in spaces_idx and wall_edges[j][0] in spaces_idx:
                curtuple = (wall_edges[i][0], wall_edges[j][0])
                distance = sPoint(G_n.nodes[curtuple[0]]['point']).distance(sPoint(G_n.nodes[curtuple[1]]['point']))
                if distance < dis_thre:
                    if curtuple not in G_n.edges.values() and (curtuple[1], curtuple[0]) not in G_n.edges.values() and curtuple[0] != curtuple[1]:
                        G_n.add_edge(curtuple[0], curtuple[1], edge_class='wall')

    # link spaces with window edges
    for i in range(len(window_edges)):
        for j in range(i+1,len(window_edges)):
            if window_edges[i][0] == window_edges[j][0] and window_edges[i][0] in windows_idx and window_edges[i][1] in spaces_idx and window_edges[j][1] in spaces_idx:
                curtuple = (window_edges[i][1], window_edges[j][1])
                if curtuple not in G_n.edges.values() and (curtuple[1], curtuple[0]) not in G_n.edges.values() and curtuple[0] != curtuple[1]:
                    G_n.add_edge(curtuple[0], curtuple[1], edge_class='window')
            if window_edges[i][0] == window_edges[j][1] and window_edges[i][0] in windows_idx and window_edges[i][1] in spaces_idx and window_edges[j][0] in spaces_idx:
                curtuple = (window_edges[i][1], window_edges[j][0])
                if curtuple not in G_n.edges.values() and (curtuple[1], curtuple[0]) not in G_n.edges.values() and curtuple[0] != curtuple[1]:
                    G_n.add_edge(curtuple[0], curtuple[1], edge_class='window')
            if window_edges[i][1] == window_edges[j][0] and window_edges[i][1] in windows_idx and window_edges[i][0] in spaces_idx and window_edges[j][1] in spaces_idx:
                curtuple = (window_edges[i][0], window_edges[j][1])
                if curtuple not in G_n.edges.values() and (curtuple[1], curtuple[0]) not in G_n.edges.values() and curtuple[0] != curtuple[1]:
                    G_n.add_edge(curtuple[0], curtuple[1], edge_class='window')
            if window_edges[i][1] == window_edges[j][1] and window_edges[i][1] in windows_idx and window_edges[i][0] in spaces_idx and window_edges[j][0] in spaces_idx:
                curtuple = (window_edges[i][0], window_edges[j][0])
                if curtuple not in G_n.edges.values() and (curtuple[1], curtuple[0]) not in G_n.edges.values() and curtuple[0] != curtuple[1]:
                    G_n.add_edge(curtuple[0], curtuple[1], edge_class='window')

    # link spaces with door edges
    for i in range(len(door_edges)):
        for j in range(i+1,len(door_edges)):
            if door_edges[i][0] == door_edges[j][0] and door_edges[i][0] in doors_idx and door_edges[i][1] in spaces_idx and door_edges[j][1] in spaces_idx:
                curtuple = (door_edges[i][1], door_edges[j][1])
                if curtuple not in G_n.edges.values() and (curtuple[1], curtuple[0]) not in G_n.edges.values() and curtuple[0] != curtuple[1]:
                    G_n.add_edge(curtuple[0], curtuple[1], edge_class='door')
            if door_edges[i][0] == door_edges[j][1] and door_edges[i][0] in doors_idx and door_edges[i][1] in spaces_idx and door_edges[j][0] in spaces_idx:
                curtuple = (door_edges[i][1], door_edges[j][0])
                if curtuple not in G_n.edges.values() and (curtuple[1], curtuple[0]) not in G_n.edges.values() and curtuple[0] != curtuple[1]:
                    G_n.add_edge(curtuple[0], curtuple[1], edge_class='door')
            if door_edges[i][1] == door_edges[j][0] and door_edges[i][1] in doors_idx and door_edges[i][0] in spaces_idx and door_edges[j][1] in spaces_idx:
                curtuple = (door_edges[i][0], door_edges[j][1])
                if curtuple not in G_n.edges.values() and (curtuple[1], curtuple[0]) not in G_n.edges.values() and curtuple[0] != curtuple[1]:
                    G_n.add_edge(curtuple[0], curtuple[1], edge_class='door')
            if door_edges[i][1] == door_edges[j][1] and door_edges[i][1] in doors_idx and door_edges[i][0] in spaces_idx and door_edges[j][0] in spaces_idx:
                curtuple = (door_edges[i][0], door_edges[j][0])
                if curtuple not in G_n.edges.values() and (curtuple[1], curtuple[0]) not in G_n.edges.values() and curtuple[0] != curtuple[1]:
                    G_n.add_edge(curtuple[0], curtuple[1], edge_class='door')

    # link spaces with direct edges
    for i in range(len(direct_edges)):
        curtuple = (direct_edges[i][0], direct_edges[i][1])
        if curtuple not in G_n.edges.values() and (curtuple[1], curtuple[0]) not in G_n.edges.values() and curtuple[0] != curtuple[1]:
            G_n.add_edge(curtuple[0], curtuple[1], edge_class='direct')

    # Recalculate the room area ratio and new attribute to rooms
    sum_area = 0
    for i in range(len(G_n.nodes)):
        if G_n.nodes[list(G_n.nodes)[i]]['class_name'] == 'rooms':
            sum_area += G_n.nodes[list(G_n.nodes)[i]]['area']
    for i in range(len(G_n.nodes)):
        if G_n.nodes[list(G_n.nodes)[i]]['class_name'] == 'rooms':
            G_n.nodes[list(G_n.nodes)[i]]['area_ratio'] = G_n.nodes[list(G_n.nodes)[i]]['area']/sum_area
        else:
            pass

    print("Simple graph--- %s seconds ---" % (time.time() - start_time))

    #### Make the multigraph
    G_m = nx.MultiGraph()
    for i in nodes:
        if nodes[i]['class_id'] == 2 or nodes[i]['class_id'] == 5 or nodes[i]['class_id'] == 0:
            G_m.add_node(i, polygon=nodes[i]['polygon'], point=nodes[i]['point'], area=nodes[i]['area'], class_name=nodes[i]['class_name'])

    # link spaces with wall edges
    for i in range(len(wall_edges)):
        for j in range(i+1,len(wall_edges)):
            if wall_edges[i][0] == wall_edges[j][0] and wall_edges[i][0] in walls_idx and wall_edges[i][1] in spaces_idx and wall_edges[j][1] in spaces_idx:
                curtuple = (wall_edges[i][1], wall_edges[j][1])
                distance = sPoint(G_m.nodes[curtuple[0]]['point']).distance(sPoint(G_m.nodes[curtuple[1]]['point']))
                if distance < dis_thre:
                    if curtuple not in G_m.edges.values() and (curtuple[1], curtuple[0]) not in G_m.edges.values() and curtuple[0] != curtuple[1]:
                        G_m.add_edge(curtuple[0], curtuple[1], edge_class='wall')
            if wall_edges[i][0] == wall_edges[j][1] and wall_edges[i][0] in walls_idx and wall_edges[i][1] in spaces_idx and wall_edges[j][0] in spaces_idx:
                curtuple = (wall_edges[i][1], wall_edges[j][0])
                distance = sPoint(G_m.nodes[curtuple[0]]['point']).distance(sPoint(G_m.nodes[curtuple[1]]['point']))
                if distance < dis_thre:
                    if curtuple not in G_m.edges.values() and (curtuple[1], curtuple[0]) not in G_m.edges.values() and curtuple[0] != curtuple[1]:
                        G_m.add_edge(curtuple[0], curtuple[1], edge_class='wall')
            if wall_edges[i][1] == wall_edges[j][0] and wall_edges[i][1] in walls_idx and wall_edges[i][0] in spaces_idx and wall_edges[j][1] in spaces_idx:
                curtuple = (wall_edges[i][0], wall_edges[j][1])
                distance = sPoint(G_m.nodes[curtuple[0]]['point']).distance(sPoint(G_m.nodes[curtuple[1]]['point']))
                if distance < dis_thre:
                    if curtuple not in G_m.edges.values() and (curtuple[1], curtuple[0]) not in G_m.edges.values() and curtuple[0] != curtuple[1]:
                        G_m.add_edge(curtuple[0], curtuple[1], edge_class='wall')
            if wall_edges[i][1] == wall_edges[j][1] and wall_edges[i][1] in walls_idx and wall_edges[i][0] in spaces_idx and wall_edges[j][0] in spaces_idx:
                curtuple = (wall_edges[i][0], wall_edges[j][0])
                distance = sPoint(G_m.nodes[curtuple[0]]['point']).distance(sPoint(G_m.nodes[curtuple[1]]['point']))
                if distance < dis_thre:
                    if curtuple not in G_m.edges.values() and (curtuple[1], curtuple[0]) not in G_m.edges.values() and curtuple[0] != curtuple[1]:
                        G_m.add_edge(curtuple[0], curtuple[1], edge_class='wall')

    # link spaces with window edges
    for i in range(len(window_edges)):
        for j in range(i+1,len(window_edges)):
            if window_edges[i][0] == window_edges[j][0] and window_edges[i][0] in windows_idx and window_edges[i][1] in spaces_idx and window_edges[j][1] in spaces_idx:
                curtuple = (window_edges[i][1], window_edges[j][1])
                if curtuple not in G_m.edges.values() and (curtuple[1], curtuple[0]) not in G_m.edges.values() and curtuple[0] != curtuple[1]:
                    G_m.add_edge(curtuple[0], curtuple[1], edge_class='window')
            if window_edges[i][0] == window_edges[j][1] and window_edges[i][0] in windows_idx and window_edges[i][1] in spaces_idx and window_edges[j][0] in spaces_idx:
                curtuple = (window_edges[i][1], window_edges[j][0])
                if curtuple not in G_m.edges.values() and (curtuple[1], curtuple[0]) not in G_m.edges.values() and curtuple[0] != curtuple[1]:
                    G_m.add_edge(curtuple[0], curtuple[1], edge_class='window')
            if window_edges[i][1] == window_edges[j][0] and window_edges[i][1] in windows_idx and window_edges[i][0] in spaces_idx and window_edges[j][1] in spaces_idx:
                curtuple = (window_edges[i][0], window_edges[j][1])
                if curtuple not in G_m.edges.values() and (curtuple[1], curtuple[0]) not in G_m.edges.values() and curtuple[0] != curtuple[1]:
                    G_m.add_edge(curtuple[0], curtuple[1], edge_class='window')
            if window_edges[i][1] == window_edges[j][1] and window_edges[i][1] in windows_idx and window_edges[i][0] in spaces_idx and window_edges[j][0] in spaces_idx:
                curtuple = (window_edges[i][0], window_edges[j][0])
                if curtuple not in G_m.edges.values() and (curtuple[1], curtuple[0]) not in G_m.edges.values() and curtuple[0] != curtuple[1]:
                    G_m.add_edge(curtuple[0], curtuple[1], edge_class='window')

    # link spaces with door edges
    for i in range(len(door_edges)):
        for j in range(i+1,len(door_edges)):
            if door_edges[i][0] == door_edges[j][0] and door_edges[i][0] in doors_idx and door_edges[i][1] in spaces_idx and door_edges[j][1] in spaces_idx:
                curtuple = (door_edges[i][1], door_edges[j][1])
                if curtuple not in G_m.edges.values() and (curtuple[1], curtuple[0]) not in G_m.edges.values() and curtuple[0] != curtuple[1]:
                    G_m.add_edge(curtuple[0], curtuple[1], edge_class='door')
            if door_edges[i][0] == door_edges[j][1] and door_edges[i][0] in doors_idx and door_edges[i][1] in spaces_idx and door_edges[j][0] in spaces_idx:
                curtuple = (door_edges[i][1], door_edges[j][0])
                if curtuple not in G_m.edges.values() and (curtuple[1], curtuple[0]) not in G_m.edges.values() and curtuple[0] != curtuple[1]:
                    G_m.add_edge(curtuple[0], curtuple[1], edge_class='door')
            if door_edges[i][1] == door_edges[j][0] and door_edges[i][1] in doors_idx and door_edges[i][0] in spaces_idx and door_edges[j][1] in spaces_idx:
                curtuple = (door_edges[i][0], door_edges[j][1])
                if curtuple not in G_m.edges.values() and (curtuple[1], curtuple[0]) not in G_m.edges.values() and curtuple[0] != curtuple[1]:
                    G_m.add_edge(curtuple[0], curtuple[1], edge_class='door')
            if door_edges[i][1] == door_edges[j][1] and door_edges[i][1] in doors_idx and door_edges[i][0] in spaces_idx and door_edges[j][0] in spaces_idx:
                curtuple = (door_edges[i][0], door_edges[j][0])
                if curtuple not in G_m.edges.values() and (curtuple[1], curtuple[0]) not in G_m.edges.values() and curtuple[0] != curtuple[1]:
                    G_m.add_edge(curtuple[0], curtuple[1], edge_class='door')

    # link spaces with direct edges
    for i in range(len(direct_edges)):
        curtuple = (direct_edges[i][0], direct_edges[i][1])
        if curtuple not in G_m.edges.values() and (curtuple[1], curtuple[0]) not in G_m.edges.values() and curtuple[0] != curtuple[1]:
            G_m.add_edge(curtuple[0], curtuple[1], edge_class='direct')

    # Recalculate the room area ratio and new attribute to rooms
    sum_area = 0
    for i in range(len(G_m.nodes)):
        if G_m.nodes[list(G_m.nodes)[i]]['class_name'] == 'rooms':
            sum_area += G_m.nodes[list(G_m.nodes)[i]]['area']
    for i in range(len(G_m.nodes)):
        if G_m.nodes[list(G_m.nodes)[i]]['class_name'] == 'rooms':
            G_m.nodes[list(G_m.nodes)[i]]['area_ratio'] = G_m.nodes[list(G_m.nodes)[i]]['area']/sum_area
        else:
            pass

    print("Multi graph--- %s seconds ---" % (time.time() - start_time))

    return G, G_n, G_m
