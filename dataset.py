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
        # load hand image and make 1ch image
        hand_path = self.dataset[idx]["hand_path"]
        hand_img = cv2.imread(hand_path) # (h, w, c)
        hand_img = hand_img[:, :, 0] # (h, w)
        hand_img = hand_img[:, :, np.newaxis] # (h, w, 1ch)

        # load tool numpy data. the number of tools = channel - 1
        tool_path = self.dataset[idx]["tool_path"]
        tool_img = np.load(tool_path) # (h, w, c)

        image = np.concatenate([hand_img, tool_img], axis=2)
        image = image.astype(np.float32) / 255.0
#         hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
        image = np.transpose(image, (2, 0, 1)) # (c, h, w)
        image = torch.tensor(image, dtype=torch.int64) # tensor

        # load label
        gaze_point = self.dataset[idx]["gaze_point"] # (x, y)
        label = torch.tensor(gaze_point, dtype=torch.float32)

        return image, label

def make_dataset():
    dataset_dicts = list()
    tool_paths = glob.glob(common.TOOL_NPY)
    tool_paths.sort()

    gaze_points = np.loadtxt(common.GAZE_CSV, delimiter=",", skiprows=1, usecols=(1, 2))
    gaze_points *= np.array([common.IMG_W, common.IMG_H]) # pix

    for i, tool_path in enumerate(tool_paths):
        if gaze_points[i][0] == 0 and gaze_points[i][1] == 0:
            continue
        hand_path = tool_path.replace("multi_channel_tool", "contour")
        hand_path = hand_path.replace("npy", "png")
        dataset_dicts.append({"hand_path" : hand_path,
                              "tool_path" : tool_path,
                              "gaze_point" : gaze_points[i],})

    return dataset_dicts


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

        cache_dict = {"dataset" : datas.dataset,
                      "length" : datas.length,}
        torch.save(cache_dict, common.DATASET_CACHE)

    return datas
