import kagglehub

# Download latest version
path = kagglehub.dataset_download("ayush1220/cifar10") + "/cifar10"

if __name__ == "__main__":
    print(path)