import cv2
import glob
import numpy as np
import os

import common

def make_images(org_paths, hand_paths, gaze_csv, inf_gaze_csv, save_dir):
    org_paths = glob.glob(org_paths)
    org_paths.sort()
    hand_paths = glob.glob(hand_paths)
    hand_paths.sort()
    gaze_points = np.loadtxt(gaze_csv, delimiter=",", skiprows=1, usecols=(1, 2))
    gaze_points *= np.array([common.INF_W, common.INF_H]) # pix
    inf_gaze_points = np.loadtxt(inf_gaze_csv, delimiter=",")
    inf_gaze_points *= np.array([common.INF_W, common.INF_H]) # pix

    for i, org_path in enumerate(org_paths):
        # load images
        org_img = cv2.imread(org_path)
        org_img = cv2.resize(org_img, (common.INF_W, common.INF_H))

        hand_img = cv2.imread(hand_paths[i])
        hand_img = cv2.resize(hand_img, (common.INF_W, common.INF_H))

        tools_path = org_path.replace("org_imgs", "tool_masks")
        tools_img = cv2.imread(tools_path)
        tools_img = cv2.resize(tools_img, (common.INF_W, common.INF_H))

        cutting_path = org_path.replace("org_imgs", "cutting_area")
        cutting_img = cv2.imread(cutting_path)
        cutting_img = cv2.resize(cutting_img, (common.INF_W, common.INF_H))

        blend_img = cv2.addWeighted(org_img, 1, tools_img, 0.4, 0)
        blend_img = cv2.addWeighted(blend_img, 1, hand_img, 0.4, 0)
        blend_img = cv2.addWeighted(blend_img, 1, cutting_img, 0.4, 0)

        # draw inferenced gaze point
        inf_gaze_point = (int(inf_gaze_points[i][0]), int(inf_gaze_points[i][1]))
        cv2.drawMarker(blend_img, inf_gaze_point, (255, 100, 50),
                       markerType=cv2.MARKER_CROSS, markerSize=50, thickness=10,
                       line_type=cv2.LINE_8) # blue

        # draw Tobii gaze point
        gaze_point = (int(gaze_points[i][0]), int(gaze_points[i][1]))
        cv2.drawMarker(blend_img, gaze_point, (0, 0, 255),
                       markerType=cv2.MARKER_CROSS, markerSize=50, thickness=10,
                       line_type=cv2.LINE_8) # red

        # put text
        # cv2.putText(blend_img, "inference", (1500, 1000), cv2.FONT_HERSHEY_SIMPLEX,
                    # 2, (0, 255, 0), thickness=2)
        # cv2.putText(blend_img, "true label", (1500, 1050), cv2.FONT_HERSHEY_SIMPLEX,
                    # 2, (0, 0, 255), thickness=2)

        # save image
        result_path = save_dir + os.path.basename(org_path)
        cv2.imwrite(result_path, blend_img)
