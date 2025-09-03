import kagglehub

def Face_detection_path():
    face_detection_path = kagglehub.dataset_download("jessicali9530/celeba-dataset")
    return face_detection_path

def Cifar10_path():
    cifar10_path = kagglehub.dataset_download("ayush1220/cifar10") + "/cifar10"
    return cifar10_path

if __name__ == "__main__":
    import sys

    paths = {
        "Face_detection" : Face_detection_path,
        "Cifar10" : Cifar10_path    
    }
    print(
        paths[sys.argv[1]]()
    )