import numpy as np
import cv2
import time

start = time.time()

hand_path = "../cutting_area_data/annotation_images/f009200.jpg"
hand_img = cv2.imread(hand_path) # (h, w, c)
hand_img = hand_img[:, :, 0]
# hand_img = hand_img.astype(np.float32) / 255.0
hand_img = hand_img[:, :, np.newaxis]

tool_path = "../cutting_area_data/annotation_images/f012400.jpg"
tool_img = cv2.imread(tool_path) # (h, w, c)
# tool_img = tool_img.astype(np.float32) / 255.0

print("t", tool_img.shape, " , h", hand_img.shape)
print("t", np.min(tool_img), np.max(tool_img), " , h", hand_img.shape, np.min(hand_img), np.max(hand_img))

image = np.concatenate([hand_img, tool_img], axis=2)
image = image.astype(np.float32) / 255.0
print(image.shape, np.min(image), np.max(image))

print("time : ", time.time() - start, "s")
