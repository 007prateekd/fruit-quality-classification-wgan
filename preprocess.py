import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from constants import *


def get_new_np_path():
    path = PATH_TO_NPY[:-4]
    new_path = path + "-" + str(IMG_SIZE) + ".npy"
    return new_path


def get_annotations(coco, all_img_info, index):
    img_file_name = all_img_info[index]["file_name"]
    ann_ids = coco.getAnnIds(all_img_info[index]["id"])
    anns = coco.loadAnns(ann_ids)
    return (img_file_name, anns)


def prepare_image(img_file_name):
    img_path = os.path.join(PATH_TO_DATA, img_file_name)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img


def prepare_label(anns):
    label = 1
    cat_ids = [ann["category_id"] for ann in anns]
    for cat_id in cat_ids:
        # 2: illness, 3: gangrene, 4: mould
        # 5: blemish, 6: dark_style_remains
        if cat_id in (2, 3, 4, 6):
            label = 0
    return label


def prepare_data(coco, all_img_info):
    new_path = get_new_np_path()
    if not os.path.isfile(new_path):
        images, labels = [], []
        for i in range(len(all_img_info)):
            (img_file_name, anns) = get_annotations(coco, all_img_info, i)
            images.append(prepare_image(img_file_name))
            labels.append(prepare_label(anns))
        images = np.array(images, dtype="float32")
        labels = np.array(labels)
        with open(new_path, "wb") as f:
            np.save(f, images)
            np.save(f, labels)
            

def id_to_label(coco, index):
    return coco.loadCats(index)[0]["name"]


def plot_segmented_img(coco, img_file_name, anns):
    img = cv2.imread(os.path.join(PATH_TO_DATA, img_file_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.axis("off")
    plt.title(img_file_name)
    plt.imshow(img)
    coco.showAnns(anns)
    plt.show()


def preprocess():
    coco = COCO(PATH_TO_ANNOT)
    all_img_ids = coco.getImgIds()
    all_img_info = coco.loadImgs(all_img_ids)
    prepare_data(coco, all_img_info)
    (img_file_name, anns) = get_annotations(coco, all_img_info, 1)
    plot_segmented_img(coco, img_file_name, anns)


if __name__ == "__main__":
    preprocess()
