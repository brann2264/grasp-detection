from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np
import sys
import json
from typing import Dict, Optional

from wilor.models import WiLoR, load_wilor
from wilor.utils import recursive_to
from wilor.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from wilor.utils.renderer import Renderer, cam_crop_to_full
from ultralytics import YOLO 
import trimesh
import pyrender
import matplotlib.pyplot as plt
LIGHT_PURPLE=(0.25098039,  0.274117647,  0.65882353)

sys.path.append(os.path.abspath("/home/brianchen/brian/PressureVision"))

# import fpn as smp
from recording.util import load_config, find_latest_checkpoint, classes_to_scalar
from recording.sequence_reader import SequenceReader
from prediction.model_builder import build_model
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import glob
from torchsummary import summary

def project_full_img(points, cam_trans, focal_length, img_res): 
    camera_center = [img_res[0] / 2., img_res[1] / 2.]
    K = torch.eye(3) 
    K[0,0] = focal_length
    K[1,1] = focal_length
    K[0,2] = camera_center[0]
    K[1,2] = camera_center[1]
    points = points + cam_trans
    points = points / points[..., -1:] 
    
    V_2d = (K @ points.T).T 
    return V_2d[..., :-1]

def create_finger_classes(force_arr, finger_points, img_shape=(1920, 1080), radius=50):

    finger_class = np.zeros(force_arr.shape)

    # points = np.array(joints_2d[finger_points])
    target_points = np.array(list(np.ndindex(img_shape)))

    differences = finger_points[np.newaxis, :, :] - target_points[:, np.newaxis, :]

    distances = np.linalg.norm(differences, axis=2)

    # closest_indices = np.argmin(distances, axis=1).reshape(temp.shape[:2])

    closest_indices = np.zeros(img_shape)

    bool_arr = (np.min(distances, axis=1) < radius).reshape(img_shape)

    closest_indices[bool_arr] = (np.argmin(distances, axis=1).reshape(img_shape)[bool_arr] + 1)/6

    finger_class[force_arr > 0] = closest_indices.T[force_arr > 0]  

    return finger_class

if __name__ == "__main__":
    model, model_cfg = load_wilor(checkpoint_path = './pretrained_models/wilor_final.ckpt' , cfg_path= './pretrained_models/model_config.yaml')
    detector = YOLO('./pretrained_models/detector.pt')
    renderer = Renderer(model_cfg, faces=model.mano.faces)
    renderer_side = Renderer(model_cfg, faces=model.mano.faces)

    device   = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model    = model.to(device)
    detector = detector.to(device)
    model.eval()

    finger_points = [4, 8, 12, 16, 20]

    config = load_config("/home/brianchen/brian/PressureVision/config/paper")
    config.TRAIN_FILTER = ["../PressureVision/data/train_fold_1/*/*"]
    config.VAL_FILTER = ["../PressureVision/data/val_fold_5/*/*"]
    config.CROP_IMAGES = False

    model_dict = build_model(config, device, ['train','val'])

    os.makedirs("finger_force_db2", exist_ok=True)
    os.makedirs("finger_force_db2/original_img", exist_ok=True)
    os.makedirs("finger_force_db2/finger_force", exist_ok=True)
    os.makedirs("finger_force_db2/hand_mesh", exist_ok=True)
    os.makedirs("finger_force_db2/finger_joints", exist_ok=True)

    for i in tqdm(range(len(model_dict["train_dataset"]))):
            # if i > 50:
            #      break
            img_path = os.path.join(model_dict["train_dataset"][i]["seq_path"], f'camera_{model_dict["train_dataset"][i]["camera_idx"]}', f'{model_dict["train_dataset"][i]["timestep"]:05}.jpg')
            img_cv2 = cv2.imread(str(img_path))
            img_shape = img_cv2.shape[:2:-1]

            detections = detector(img_cv2, conf = 0.3, verbose=False)[0]
            bboxes    = []
            is_right  = []
            for det in detections: 
                Bbox = det.boxes.data.cpu().detach().squeeze().numpy()
                is_right.append(det.boxes.cls.cpu().detach().squeeze().item())
                bboxes.append(Bbox[:4].tolist())
            
            if len(bboxes) == 0:
                continue
            boxes = np.stack(bboxes)
            right = np.stack(is_right)
            dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=2.0)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

            all_verts = []
            all_cam_t = []
            all_right = []
            all_joints= []
            all_kpts  = []
            
            for batch in dataloader: 
                batch = recursive_to(batch, device)
        
                with torch.no_grad():
                    out = model(batch) 
                    
                multiplier    = (2*batch['right']-1)
                pred_cam      = out['pred_cam']
                pred_cam[:,1] = multiplier*pred_cam[:,1]
                box_center    = batch["box_center"].float()
                box_size      = batch["box_size"].float()
                img_size      = batch["img_size"].float()
                scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
                pred_cam_t_full     = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

                
                # Render the result
                batch_size = batch['img'].shape[0]
                for n in range(batch_size):
                    # Get filename from path img_path
                    # img_fn, _ = os.path.splitext(os.path.basename(img_path))
                    
                    verts  = out['pred_vertices'][n].detach().cpu().numpy()
                    joints = out['pred_keypoints_3d'][n].detach().cpu().numpy()
                    
                    is_right    = batch['right'][n].cpu().numpy()
                    verts[:,0]  = (2*is_right-1)*verts[:,0]
                    joints[:,0] = (2*is_right-1)*joints[:,0]
                    cam_t = pred_cam_t_full[n]
                    kpts_2d = project_full_img(verts, cam_t, scaled_focal_length, img_size[n])
                    joints_2d = project_full_img(joints, cam_t, scaled_focal_length, img_size[n])
                    
                    all_verts.append(verts)
                    all_cam_t.append(cam_t)
                    all_right.append(is_right)
                    all_joints.append(joints)
                    all_kpts.append(kpts_2d)
                    
                    # Save all meshes to disk
                    camera_translation = cam_t.copy()
                    tmesh = renderer.vertices_to_trimesh(verts, camera_translation, LIGHT_PURPLE, is_right=is_right)
                    # tmesh.export(os.path.join("finger_force_db", f'{img_fn}_{n}.obj'))

                    force_arr = model_dict["train_dataset"][i]["seq_reader"].get_force_pytorch(model_dict["train_dataset"][i]["camera_idx"], model_dict["train_dataset"][i]["timestep"], config)
                    
                    finger_classes = np.zeros((1080, 1920, 3)) * 255
                    finger_classes[:, :, 0] = create_finger_classes(force_arr, np.array(joints_2d[finger_points]), radius=50)
                    
                    # temp[:, :, 0] = create_finger_classes(force_arr, np.array(joints_2d[finger_points]))

                    # cv2.imwrite(os.path.join("finger_force_db/finger_force", f'force_{img_fn}.jpg'), finger_classes)
                    np.save(os.path.join("finger_force_db2/finger_force", f'force_{i}.jpg'), finger_classes)
                    np.save(os.path.join("finger_force_db2/finger_joints", f"{i}"), joints_2d[finger_points])

            # Render front view
            if len(all_verts) > 0:
                misc_args = dict(
                    mesh_base_color=LIGHT_PURPLE,
                    scene_bg_color=(1, 1, 1),
                    focal_length=scaled_focal_length,
                )
                cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[n], is_right=all_right, **misc_args)

                # Overlay image
                # input_img = img_cv2.astype(np.float32)[:,:,::-1]/255.0
                # input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
                # input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]
                # print(cam_view)
                # print(255*cam_view[:, :, ::-1])
                cv2.imwrite(os.path.join("finger_force_db2/original_img", f'{i}.jpg'), img_cv2)
                cv2.imwrite(os.path.join("finger_force_db2/hand_mesh", f'{i}.jpg'), 255*cam_view[:, :, ::-1])