#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@Time    : 2020-02-25 11:54
@Author  : Jann
@Contact : l33klin@foxmail.com
@Site    : 
@File    : data_preprocess.py
"""
import os
from PIL import Image
from utils import image_cutter

user = 'klin'

raw_path = "/Users/{}/Nextcloud/Documents/MineCraft/2020-02-26/raw_dataset".format(user)
dest_path = "/Users/{}/Nextcloud/Documents/MineCraft/2020-02-26/dataset".format(user)

area_ratio = {
    "left": 13 / 32,
    "right": 21 / 36,
    "top": 4 / 20,
    "bottom": 19 / 32
}


def walk_path(path):
    print("Raw path: {}".format(path))
    for root, dirs, files in os.walk(path, topdown=True):
        print("root: ", root)
        for name in dirs:
            raw = os.path.join(root, name)
            dest = os.path.join(root.replace(raw_path, dest_path), name)
            if not os.path.exists(dest):
                os.mkdir(dest)
            print("{} -> {}".format(raw, dest))
        for name in files:
            if name.startswith('.') or not name.endswith(".jpg"):
                continue
            raw = os.path.join(root, name)
            dest = os.path.join(root.replace(raw_path, dest_path), name)
            img = Image.open(raw)
            cut_img = image_cutter(img, area_ratio)
            cut_img.save(dest)
            print("{} -> {}".format(raw, dest))


walk_path(raw_path)
