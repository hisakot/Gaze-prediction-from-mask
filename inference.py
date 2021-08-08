import argparse
import cv2
import glob
import numpy as np
import os

import torch

import common
import blend
import model


def forward(model, device, hand_path):
    # load hand image
    hand_img = cv2.imread(hand_path)
    hand_img = hand_img[:, :, 0] #(h, w)
    hand_img = hand_img[:, :, np.newaxis] # (h, w, 1ch)

    # load tool image
    tool_path = hand_path.replace("contour", "multi_channel_tool")
    tool_path = tool_path.replace("png", "npy")
    tool_img = cv2.imread(tool_path)
#     tool_img = tool_img[:, :, 1] # when tool_masks

    # concatenate images
    image = np.concatenate([hand_img, tool_img], axis=2)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1)) # (c, h, w)

    # make single batch
    image = torch.tensor(image[np.newaxis, :, :], dtype=torch.int64) # tensor
    image = image.to(device)
    with torch.no_grad():
        output, out = model(image) # (x, y)

    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--network", type=string, required=True,
                        help="choose network, only ResNet50 or ResNet50LSTM")
    parser.add_argument("--input_ch", type=int, required=True,
                        help="the number of input channel")
    parser.add_argument("--best_model", default="model.pth", required=True,
                        help="choose using model you think best")
    args = parser.parse_args()

    # setup model and GPU
    if args.network == "ResNet50":
        model = ResNet50(pretrained=False, num_input_channel=args.input_ch,
                         num_output=2)
    elif args.network == "ResNet50LSTM":
        model = model.ResNet50LSTM(pretrained=False,
                                   num_input_channel=args.input_ch, num_output=2)
    model, device = common.setup_device(model)

    save_model = glob.glob(common.MODEL_SAVE_PATH + args.best_model)[0]
    print(save_model)
    checkpoint = torch.load(save_model, map_location=device)
    if torch.cuda.device_count() >= 1:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        from collections import OrderedDict
        state_dict = OrderedDict()
        for k, v in checkpoint["model_state_dict"].items():
            name = k[7:] # remove "module."
            state_dict[name] = v
        model.load_state_dict(state_dict)
    model.eval()

    # main
    hand_paths = glob.glob(common.TEST_HAND_IMG)
    hand_paths.sort()
    org_paths = glob.glob(common.TEST_ORG_IMG)
    org_paths.sort()

    if not os.path.exists(INF_GAZE_CSV):
        print("========== forward and save inferenced gaze points as csv ==========")
        for i, hand_path in enumerate(hand_paths):
            # inference
            output = forward(model, device, hand_path)
            output = output.cpu().numpy()

            # save inferenced gaze as txt
            with open (INF_GAZE_CSV, "a") as f:
                np.savetxt(f, output / np.array([common.INF_W, common.INF_H]), delimiter=",")

    print("========== save images ==========")
    blend.make_images(common.TEST_ORG_IMG, common.TEST_HAND_IMG, common.TEST_GAZE_CSV, common.INF_GAZE_CSV, common.RESULT_DIR)
