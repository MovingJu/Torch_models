import os
import cv2
import torch
from torch.utils.data import Dataset
import pandas as pd

class Datasets(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 이미지 경로
        img_path = os.path.join(self.image_dir, self.data.iloc[idx, 0])
        img = cv2.imread(img_path)  # (H, W, C), BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Grayscale: (H, W)

        # Resize to (96, 96)
        img = cv2.resize(img, (96, 96))

        # Normalize to 0~1 and add channel dimension
        img = img.astype('float32') / 255.0
        img = torch.from_numpy(img).unsqueeze(0)  # (1, 96, 96)

        keypoints = self.data.iloc[idx, 1:].values.astype('float32')
        keypoints = torch.tensor(keypoints)

        return img, keypoints
