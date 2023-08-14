import os
from pathlib import Path

import numpy as np
from PIL import Image

from util.preprocess_itw_im import preprocessInTheWildImage

img_list = []


def get_all_img_files(rootDir: Path, valid_ext: list):
    for lists in os.listdir(rootDir):
        path = os.path.join(rootDir, lists)
        filename, file_extension = os.path.splitext(path)
        if file_extension in valid_ext and "parsings" not in path:
            img_list.append(path)
        if os.path.isdir(path):
            get_all_img_files(path, valid_ext)


def make_parsings(preprocessor: preprocessInTheWildImage):
    """
    For each image in the dataset, we want a segmentation map of where the different landmarks of the face like eyes,
    nose etc. are and store it as an image. These are called as parsings in the context of the paper and the dataset.
    """
    count = 0

    for img_file in img_list:
        img = Image.open(img_file).convert('RGB')
        img = np.array(img.getdata(), dtype=np.uint8).reshape(
            img.size[1], img.size[0], 3
        )

        _, parsing = preprocessor.forward(img)
        if parsing is None:
            print("Skipped: " + img_file)
            continue

        # Construct the new file path with the updated name
        root = img_file.split("/")[0:-1]
        root = "/".join(root)
        img_name = img_file.split("/")[-1]

        parsings_path = os.path.join(root, "parsings", img_name[:-4] + ".png")

        # Save parsing in parsings_path directory
        parsing = Image.fromarray(parsing)
        parsing.save(parsings_path)

        count += 1

        if count % 10000 == 0:
            print(f"{count} done")

    print(count)


def main():
    preprocessor = preprocessInTheWildImage(out_size=256)

    valid_ext = [".jpg"]

    get_all_img_files(Path("/slingshot/session/datasets/datasets"), valid_ext)

    print("Got the list of images")

    make_parsings(preprocessor)


if __name__ == '__main__':
    main()
