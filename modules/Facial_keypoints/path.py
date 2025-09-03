import kagglehub

# Download latest version
path = kagglehub.dataset_download("jessicali9530/celeba-dataset")

if __name__ == "__main__":
    print(path)