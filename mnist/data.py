import torchvision

OUTPUT_PATH = "/mnt/output"


if __name__ == '__main__':
    # Let's download the MNIST dataset and save it to the output directory so that an artifact is created
    torchvision.datasets.MNIST(OUTPUT_PATH, download=True, train=True)
