import torch
import torch.nn as nn
import torchvision.transforms as transforms


class Facial_keypoints(nn.Module):
    def __init__(self) -> None:
        super(Facial_keypoints, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.cnn3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.flatten = nn.Flatten()
        self.nn1 = nn.Sequential(
            nn.Linear(128 * 27 * 22, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )
    def forward(self, x):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.flatten(x)
        x = self.nn1(x)
        return x
    
    @staticmethod
    def set_device():
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    @staticmethod
    def main():
        import os
        from torch.utils.data.dataloader import DataLoader
        from Datasets import Datasets
        import path

        device = Facial_keypoints.set_device()

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        full_datas = Datasets(path.path, "list_bbox_celeba.csv", "img_align_celeba/img_align_celeba", transform=transform)
        model = Facial_keypoints().to(device)
        criterion = nn. MSELoss()
        optimizer = 1e-3
        
        train_dataset, test_dataset = torch.utils.data.random_split(full_datas, [0.8, 0.2])
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=(os.cpu_count() or 4))

        total_epoch = 10

        for epoch in range(total_epoch):
            model.train()
            total_loss = 0
            # correct = 0
            # total = 0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                loss.backward()

                total_loss += loss.item()
                # preds = outputs.argmax(dim=1)
                # correct += (preds == batch_y).sum().item()
                # total += batch_y.size(0)
            
            # acc = correct / total * 100
            # print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}, Accuracy: {acc:.2f}%")
            print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}")

        torch.save(model, "FK_test_model.pt")

        return




if __name__ == "__main__":
    Facial_keypoints.main()
