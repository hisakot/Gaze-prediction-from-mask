import torch
import torch.nn as nn

IMG_W = 320
IMG_H = 180

DATASET_CACHE = "./dataset_cache"
ORG_IMG = "../main20180525/org_imgs/*.png"
HAND_IMG = "../main20180525/contour/*.png"
TOOL_NPY = "../main20180525/multi_channel_tool/*.npy"
TOOL_IMG = "../main20180525/tool_masks/*.png"
CUT_IMG = "../main20180525/cutting_area/*.png"
GAZE_CSV = "../main20180525/gaze.csv"

MODEL_SAVE_PATH = "./models/"

INF_W = 960
INF_H = 540

TEST_ORG_IMG = "../main20170707/org_imgs/*.png"
MRSATO_IMG = "../main20170707/mrSato_output/*.png"
TEST_HAND_IMG = "../main20170707/contour/*.png"
TEST_TOOL_IMG = "../main20170707/tool_masks/*.png"
TEST_CUT_IMG = "../main20170707/cutting_area/*.png"

TEST_GAZE_CSV = "../main20170707/gaze.csv"
INF_GAZE_CSV = "../main20170707/inf_gaze_from_hand_tool_cutting_area_3ch_resnet18lstm_new10.csv"

RESULT_DIR = "../main20170707/result/"

def setup_device(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 0:
        print("---------- Use ", torch.cuda.device_count(), "GPUs ----------")
        model = nn.DataParallel(model)
    else:
        print("---------- Use CPU ----------")
    model.to(device)

    return model, device
