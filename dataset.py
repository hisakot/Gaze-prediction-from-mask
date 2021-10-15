import csv
import cv2
import glob
import numpy as np
import os

import torch
from torch.utils.data import Dataset

import common

class Datas():
    def __init__(self):
        self.dataset = list()
        self.length = 0

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_list = list()
        target_list = list()
        for i in range(25):
            # load hand image
            hand_path = self.dataset[idx+i]["hand_path"]
            hand_img = cv2.imread(hand_path) # (h, w, c)
            hand_img = cv2.resize(hand_img, (common.IMG_W, common.IMG_H))
            hand_img = hand_img[:, :, 0] # (h, w)
            hand_img = hand_img[:, :, np.newaxis] # (h, w, 1ch)
    
            # load tool image
            tool_path = self.dataset[idx+i]["tool_path"]
            tool_img = cv2.imread(tool_path) # (h, w, c)
            tool_img = cv2.resize(tool_img, (common.IMG_W, common.IMG_H))
            tool_img = tool_img[:, :, 1] # (h, w)
            tool_img = tool_img[:, :, np.newaxis] # (h, w, 1ch)
    
            # load cutting_area imag
            cutting_path = self.dataset[idx+i]["cutting_path"]
            cutting_img = cv2.imread(cutting_path) # (h, w, c)
            cutting_img = cv2.resize(cutting_img, (common.IMG_W, common.IMG_H))
            cutting_img = cutting_img[:, :, 0] # (h, w)
            cutting_img = cutting_img[:, :, np.newaxis] # (h, w, 1ch)
    
            # concatenate images
            image = np.concatenate([hand_img, tool_img], axis=2)
            image = np.concatenate([image, cutting_img], axis=2)
            image = image.astype(np.float32) / 255.0
            image = np.transpose(image, (2, 0, 1)) # (c, h, w)
            image = torch.tensor(image, dtype=torch.int64) # tensor
            image_list.append(image)
    
    #         org_path = self.dataset[idx+i]["org_path"]
    #         org_img = cv2.imread(org_path) # (h, w, c)
    #         org_img = cv2.resize(org_img, (common.IMG_W, common.IMG_H))
    #         image = np.transpose(org_img, (2, 0, 1)) # (c, h, w)
    #         image = torch.tensor(image, dtype=torch.int64) # tensor

            # load label
            gaze_point = self.dataset[idx+i]["gaze_point"] # (x, y)
            label = torch.tensor(gaze_point, dtype=torch.float32)
            target_list.append(label)
        image_list = torch.stack(image_list, dim=0)
        target_list = torch.stack(target_list, dim=0)

        return image_list, target_list

def make_dataset():
    dataset_dicts = list()
    org_paths = glob.glob(common.ORG_IMG)
    org_paths.sort()

    gaze_points = np.loadtxt(common.GAZE_CSV, delimiter=",", skiprows=1, usecols=(1, 2))
    gaze_points *= np.array([common.IMG_W, common.IMG_H]) # pix

    for i, org_path in enumerate(org_paths):
        if gaze_points[i][0] == 0 and gaze_points[i][1] == 0:
            continue
        hand_path = org_path.replace("org_imgs", "contour")
        tool_path = org_path.replace("org_imgs", "tool_masks")
        cutting_path = org_path.replace("org_imgs", "cutting_area")
        dataset_dicts.append({"org_path" : org_path,
                              "hand_path" : hand_path,
                              "tool_path" : tool_path,
                              "cutting_path" : cutting_path,
                              "gaze_point" : gaze_points[i],})

    return dataset_dicts

def make_temporal_data(data, frame_num):
    image = list()
    target = list()
    for idx in range(frame_num):
        image.append(data[idx][0])
        target.append(data[idx][1])
    image = torch.stack(image, dim=0)
    return image, target

def setup_data():
    datas = Datas()

    try:
        cache = torch.load(common.DATASET_CACHE)
        datas.dataset = cache["dataset"]
        datas.length = cache["length"]

    except FileNotFoundError:
        dataset_dicts = make_dataset()
        datas.dataset = dataset_dicts
        datas.length = len(datas.dataset)
        print(datas.length)

        cache_dict = {"dataset" : datas.dataset,
                      "length" : datas.length,}
        torch.save(cache_dict, common.DATASET_CACHE)

    return datas
