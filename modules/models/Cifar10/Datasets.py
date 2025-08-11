import os
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import read_image

class Datasets(Dataset):
    def __init__(self, img_dir, train=True, transform=None, target_transform=None):
        self.mode = "train" if train else "test"
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

        self.classes = sorted(entry.name for entry in os.scandir(os.path.join(img_dir, self.mode)) if entry.is_dir())
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.samples = []
        for cls_name in self.classes:
            class_dir = os.path.join(img_dir, self.mode, cls_name)
            for entry in os.scandir(class_dir):
                if entry.is_file() and entry.name.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append((entry.path, self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> tuple[Tensor, int]:
        img_path, label = self.samples[idx]
        image = read_image(img_path).float() / 255.0
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


if __name__ == "__main__":
    print("test it in /modules/models/Cifar10/test.ipynb")