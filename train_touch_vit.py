import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

EPOCHS = 20
VERSION = 2.1

class force_db(Dataset):

    def __init__(self, img_path, force_path, fingers_path, transforms=torchvision.models.ViT_B_32_Weights.IMAGENET1K_V1.transforms(), padding=20, upper=-1, lower=0):
        img_paths = os.listdir(img_path)
        img_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        force_paths = os.listdir(force_path)
        force_paths.sort(key=lambda x: int((os.path.basename(x)).split(".")[0].split("_")[1]))
        finger_paths = os.listdir(fingers_path)
        finger_paths.sort(key= lambda x: int(os.path.splitext(os.path.basename(x))[0]))

        self.img_paths = [os.path.join(img_path, path) for path in img_paths][lower:upper]
        self.force_paths = [os.path.join(force_path, path) for path in force_paths][lower:upper]
        self.finger_paths = [os.path.join(fingers_path, path) for path in finger_paths][lower:upper]

        self.force_arrs = [np.load(path).astype(np.float16) for path in tqdm(self.force_paths)]
        self.img_arrs = [cv2.imread(path).astype(np.float16) for path in self.img_paths]
        self.finger_idxs = [np.load(path).astype(np.float16) for path in self.finger_paths]

        self.padding = padding
        self.transforms = transforms
    
    def __getitem__(self, index):
        force_arr = self.force_arrs[index]
        img = self.img_arrs[index]
        finger_idxs = self.finger_idxs[index].astype(int)[:, ::-1]

        force_vals = np.sort(np.unique(force_arr))

        if len(force_vals) <= 1:
            finger = 1 #np.random.randint(0, len(finger_idxs))
            x_min, y_min = finger_idxs[finger] - self.padding - 80
            x_max, y_max = finger_idxs[finger] + self.padding + 80
            x_min, y_min = max(x_min,0), max(y_min,0)
            x_max, y_max = min(x_max, img.shape[0]) , min(y_max, img.shape[1])
            # print(finger_idxs[finger])
            # print(img[x_min:x_max, y_min:y_max].shape)

            return self.transforms(torch.tensor(img[x_min:x_max, y_min:y_max].transpose(2, 0, 1))), np.float32(0.0)
        
        finger = 1 #np.random.randint(1, len(np.unique(force_arr)))
        indices = np.argwhere(force_arr == force_vals[finger])
        
        x_min, y_min, _ = indices.min(axis=0) - self.padding
        x_max, y_max, _ = indices.max(axis=0) + self.padding
        x_min, y_min = max(x_min,0), max(y_min,0)
        x_max, y_max = min(x_max, img.shape[0]) , min(y_max, img.shape[1])

        return self.transforms(torch.tensor(img[x_min:x_max, y_min:y_max].transpose(2, 0, 1))), np.float32(1.0)

    def __len__(self):
        return len(self.finger_idxs)

if __name__ == "__main__":
    model = torchvision.models.vit_b_32(weights=torchvision.models.ViT_B_32_Weights.DEFAULT)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model.heads = nn.Sequential(nn.Linear(768, 256), 
                                nn.GELU(),
                                nn.BatchNorm1d(256),
                                nn.Linear(256, 1))
    model = model.to(device)

    train_dataset = force_db("finger_force_db/original_img", "finger_force_db/finger_force", "finger_force_db/finger_joints",upper=2300)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataset = force_db("finger_force_db/original_img", "finger_force_db/finger_force", "finger_force_db/finger_joints",lower=2300, upper=3100)
    val_loader = DataLoader(val_dataset, batch_size=32)

    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    LRStepper = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.3)

    for epoch in tqdm(range(EPOCHS)):
        model.train()  # Set model to training mode
        epoch_loss = 0
        acc = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            # pred = torch.round(torch.sigmoid(outputs))
            loss = loss_fn(outputs.flatten(), targets)
            pred = torch.sigmoid(outputs).flatten() > 0.5
            acc += (pred==targets).sum()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            LRStepper.step()

            epoch_loss += loss.item()

        # Print epoch loss
        print(f"Epoch {epoch + 1}/{EPOCHS}, Train Loss: {epoch_loss / len(train_dataset)}, Train Acc: {acc/len(train_dataset)}")

        model.eval()
        val_loss = 0
        val_acc = 0
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            pred = torch.round(torch.sigmoid(outputs))
            loss = loss_fn(outputs.flatten(), targets)
            val_acc += (pred.flatten() == targets).sum()

            val_loss += loss.item()
        
        print(f"Epoch {epoch + 1}/{EPOCHS}, Val Loss: {val_loss / len(val_dataset)}, Val Acc: {val_acc / len(val_dataset)}")

        if epoch % 2 == 0:
            torch.save(model.state_dict(), f'touch_vit_models/model_weights{VERSION}_ckpt{epoch}.pth')
    
    torch.save(model.state_dict(), f'touch_vit_models/model_weights{VERSION}_complete.pth')