import os
import glob
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torchvision
import torch
import torch.nn as nn

data_num = int(sys.argv[1])

if sys.argv[2] == "Test":
    contact = "-"
elif sys.argv[2] == "True":
    contact = "contact"
else:
    contact = "no_contact"

skip_frames = 5


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FPS, 30)

    window_name = "Image Frame"
    disp_x = 480 * 2
    disp_y = 384 * 2
    aspect_ratio = 480 / 384
    disp_frame = np.zeros((disp_y, disp_x, 3), dtype=np.uint8)
    cv2.imshow(window_name, disp_frame)

    frame = 0

    while True:

        for _ in range(skip_frames):
            cap.grab()

        ret, camera_frame = cap.read()

        if camera_frame is None:
            continue

        rgb_frame = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)

        if contact != "-":
            cv2.imwrite(f"./touch_dataset_new/{contact}/sample_{data_num}_{frame}.jpeg", rgb_frame)

        cv2.imshow(window_name, camera_frame)
        frame += 1

        keycode = cv2.waitKey(1) & 0xFF

        if keycode == ord('q'):
            break    

    cv2.destroyAllWindows()