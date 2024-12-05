import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torchvision
import torch
import torch.nn as nn

vit_transforms = torchvision.models.ViT_B_32_Weights.IMAGENET1K_V1.transforms()

def locate_index(detection_result, hand):
    x = detection_result.hand_landmarks[hand][8].x * 1920
    y = detection_result.hand_landmarks[hand][8].y * 1080
    coords = np.array((x, y)).astype(int)
    x_min, y_min = coords - 20 - 80
    x_max, y_max = coords + 20 + 80
    x_min, y_min = max(x_min,0), max(y_min,0)
    x_max, y_max = min(x_max, 1920) , min(y_max, 1080)

    return x_min, x_max, y_min, y_max

def process_img(img, transforms=vit_transforms):
    return transforms(torch.tensor(img.transpose((2, 0, 1))))[np.newaxis]


if __name__ == "__main__":
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options,
                                        num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)

    model = torchvision.models.vit_b_32(weights=torchvision.models.ViT_B_32_Weights.DEFAULT)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model.heads = nn.Sequential(nn.Linear(768, 256), 
                                nn.GELU(),
                                nn.BatchNorm1d(256),
                                nn.Linear(256, 1))
    model = model.to(device)

    model.load_state_dict(torch.load("touch_vit_models/model_weights3.1_complete.pth", map_location=torch.device('cpu')))

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FPS, 30)

    window_name = "Touch prediction"
    disp_x = 480 * 2
    disp_y = 384 * 2
    aspect_ratio = 480 / 384
    disp_frame = np.zeros((disp_y, disp_x, 3), dtype=np.uint8)
    cv2.imshow(window_name, disp_frame)

    model.eval()
    text_x, text_y = (0,0)

    while True:
        ret, camera_frame = cap.read()
        if camera_frame is None:
            continue
        
        rgb_frame = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        hand_landmarks = detector.detect(img)

        if len(hand_landmarks.hand_landmarks) > 0:

            for i in range(len(hand_landmarks.hand_landmarks)):

                x_min, x_max, y_min, y_max = locate_index(hand_landmarks, i)

                input = process_img(camera_frame[y_min:y_max, x_min:x_max])
                output = model(input)
                pred = torch.round(torch.sigmoid(output))

                text = "Touch" if pred == 0 else "No Contact"
                color = (255, 0, 0) if pred == 0 else (0, 0, 255)

                cv2.rectangle(camera_frame, (x_min, y_min), (x_max, y_max), color, 2)            

                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                text_x = x_min
                text_y = y_min - 10  # Position text above the bounding box
                cv2.rectangle(
                    camera_frame,
                    (text_x, text_y - text_size[1] - 4),  # Text background
                    (text_x + text_size[0] + 4, text_y + 4),
                    color,
                    -1,
                )
                cv2.putText(
                    camera_frame,
                    text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

        cv2.imshow(window_name, camera_frame)
        keycode = cv2.waitKey(1) & 0xFF

        if keycode == ord('q'):
            break    

    cv2.destroyAllWindows()
    

