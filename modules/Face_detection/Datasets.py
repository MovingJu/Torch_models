import torchvision.transforms as transform
from torch.utils.data import Dataset
import torch, cv2, pandas as pd
import numpy as np

class Datasets(Dataset):
    def __init__(self, dataset_path: str, csv_file_name: str, image_path: str, is_train: bool = False, transform: transform.Compose | None = None):
        self.dataset_path = dataset_path
        self.data = pd.read_csv(dataset_path + "/" + csv_file_name)
        self.image_dir = dataset_path + "/" + image_path

        self.labels = []
        self.images = []
        for idx, _ in enumerate(self.data):
            img = cv2.imread(
                        self.image_dir + "/" + str(self.data.iloc[idx, 0]), 
                        cv2.IMREAD_COLOR
                    )
            
            if img is None:
                raise RuntimeError("Image is not valid!")
            if transform:
                img = transform(img)

            self.images.append(img)
            self.labels.append(
                torch.tensor([
                        self.data.iloc[idx, 1],
                        self.data.iloc[idx, 2],
                        self.data.iloc[idx, 3],
                        self.data.iloc[idx, 4]
                    ],
                    dtype=torch.float
                )
            )
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        temp = self.images[idx]
        label = self.labels[idx]

        return temp, label

if __name__ == "__main__":
    import modules

    data = Datasets(modules.path.Face_detection_path(), "list_bbox_celeba.csv", "img_align_celeba/img_align_celeba")

    img, label = data[100]

    print(f"shape: {img.shape}, label: {label}")

    cv2.imshow("Original", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()