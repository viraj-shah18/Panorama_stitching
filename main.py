import yaml
import cv2
from typing import Any, List
import os


def load_images(name_list: List[str]) -> List[Any]:
    images = []
    for fname in name_list:
        img = cv2.imread(fname)
        images.append(img)
    return images


def save_img(img):
    cv2.imwrite(SAVE_PATH, img)


if __name__ == "__main__":
    CWD = os.getcwd()
    CONFIG_PATH = os.path.join(CWD, "config.yaml")
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    fnames = config.get("fname_images", list())
    num_images = len(fnames)
    use_depth = config.get("use_depth", False)
    SAVE_FOLDER = config.get("save_path", CWD)
    SAVE_PATH = os.path.join(SAVE_FOLDER, f"stitch_{num_images}imgs.jpg")
    if not os.path.exists(SAVE_FOLDER):
        os.mkdir(SAVE_FOLDER)

    if num_images < 2:
        raise ValueError("Provide at least 2 images to stitch in config file")
    elif num_images > 4:
        raise NotImplementedError("Currently it only supports upto 4 images")

    if use_depth:
        from rgbd_img import stitch
    else:
        from rgb_img import stitch
    images = load_images(fnames)
    print("Stitching first 2 images")
    img12 = stitch(images[0], images[1], 400, 5)

    if num_images == 2:
        cv2.imwrite(SAVE_PATH, img12)

    elif num_images == 4:
        print("Stitching last 2 images")
        img34 = stitch(images[2], images[3], 400, 5)
        print("Stitching 2 created images")
        img1234 = stitch(img12, img34, 400, 15)
        cv2.imwrite(SAVE_PATH, img1234)
    else:
        print("Stitching the last image to stitched image")
        img123 = stitch(img12, images[2], 400, 5)
        cv2.imwrite(SAVE_PATH, img123)
