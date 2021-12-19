import numpy as np
import yaml
import cv2
from typing import Any, List
import os
import matplotlib.pyplot as plt


def load_rgb_images(name_list: List[str]) -> List[Any]:
    images = []
    for fname in name_list:
        img = cv2.imread(fname)
        images.append(img)
    return images

def load_rgbd_images(srcs, refs):
    src_img = cv2.imread(srcs['img_file'])
    src_depth = cv2.imread(srcs['depth_file'], 0)

    ref_img = cv2.imread(refs['img_file'])
    ref_depth = cv2.imread(refs['depth_file'], 0)
    images = [src_img, ref_img]
    return images, src_depth, ref_depth

def save_img(save_path, img, use_depth=False, src_img=None):
    if not use_depth:
        cv2.imwrite(save_path, img)
        return
    
    # using src image and warp images together
    concat_img = np.concatenate((src_img, img), axis=0)
    # concat_img = cv2.vconcat([src_img, img])
    cv2.imwrite(save_path, concat_img)
    

if __name__ == "__main__":
    CWD = os.getcwd()
    CONFIG_PATH = os.path.join(CWD, "config.yaml")
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    use_depth = config.get("use_depth", False)
    SAVE_FOLDER = config.get("save_path", CWD)
    if not os.path.exists(SAVE_FOLDER):
        os.mkdir(SAVE_FOLDER)
    
    if use_depth:
        from rgbd_img import stitch
        SAVE_PATH = os.path.join(SAVE_FOLDER, f"warped_compare.jpg")
        srcs, refs = config.get("src_image", None), config.get("ref_image", None)
        if srcs is None or refs is None:
            raise ValueError("Incorrect Arguments in config file")
        
        images, src_depth, ref_depth = load_rgbd_images(srcs, refs)
        num_images=2

    else:
        from rgb_img import stitch
        fnames = config.get("fname_images", list())
        num_images = len(fnames)
        SAVE_PATH = os.path.join(SAVE_FOLDER, f"stitch_{num_images}imgs.jpg")
        if num_images < 2:
            raise ValueError("Provide at least 2 images to stitch in config file")
        elif num_images > 4:
            raise NotImplementedError("Currently it only supports upto 4 images")
        
        images = load_rgb_images(fnames)
        src_depth = None
        ref_depth = None
    
    print("Stitching first 2 images")
    img12 = stitch(images[0], images[1], src_depth=src_depth, ref_depth=ref_depth, built_in=False)

    if num_images == 2:
        save_img(SAVE_PATH, img12, use_depth, images[0])

    elif num_images == 4:
        print("Stitching last 2 images")
        img34 = stitch(images[2], images[3])
        print("Stitching 2 created images")
        img1234 = stitch(img12, img34, threshold=15)
        save_img(SAVE_PATH, img1234)
    else:
        print("Stitching the last image to stitched image")
        img123 = stitch(img12, images[2])
        save_img(SAVE_PATH, img123)
