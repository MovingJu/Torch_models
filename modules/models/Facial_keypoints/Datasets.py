import torch
from torch.utils.data import Dataset
import cv2, pandas as pd

class Datasets(Dataset):
    def __init__(self, dataset_path: str, csv_file_name: str, image_path: str, transform=None):
        self.dataset_path = dataset_path
        self.data = pd.read_csv(dataset_path + "/" + csv_file_name)
        self.image_dir = dataset_path + "/" + image_path
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label: tuple = (
            self.data.iloc[idx, 1],
            self.data.iloc[idx, 2],
            self.data.iloc[idx, 3],
            self.data.iloc[idx, 4],
        )
        file_name = self.image_dir + "/" + str(self.data.iloc[idx, 0])
        img = cv2.imread(file_name, cv2.IMREAD_COLOR)

        if type(img) == None:
            test = cv2.Mat(1)
            return test, -1

        return img, label

if __name__ == "__main__":
    from path import path

    data = Datasets(path, "list_bbox_celeba.csv", "images")
    img, label = data[1]
    # if not img:
    #     print("Exit cuz image is not valid")
    #     exit()
    print(f"shape: {img.shape}, label: {label}")

    # cv2.imshow("Original", img)
    
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()