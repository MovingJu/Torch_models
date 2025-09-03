from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transforms
import torch
import os

import modules

def main():
    device = modules.Face_detection_model.set_device()

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    full_datas = modules.Datasets(modules.path.Face_detection_path(), "list_bbox_celeba.csv", "img_align_celeba/img_align_celeba", transform=transform)
    model = modules.Face_detection_model().to(device)
    criterion = torch.nn.MSELoss()
    optimizer = 1e-3
    print("loaded datas")
    
    train_dataset, test_dataset = torch.utils.data.random_split(full_datas, [0.8, 0.2])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=(os.cpu_count() or 4))

    total_epoch = 0

    print("training start!")
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

    print("model entered test section!")
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=(os.cpu_count() or 4))
    with torch.no_grad():
        model.eval()
        final_total_loss = 0
        final_loss = 0
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            output = model(batch_x)
            final_loss = criterion(output, batch_y)
            final_total_loss += final_loss

    print("saving model...")
    torch.save(model, f"./modules/Face_detection/model/FD_test_model_{final_total_loss: .4f}.pt")

    print("-" * 6 + "Complete!" + "-" * 6)

    return




if __name__ == "__main__":
    main()