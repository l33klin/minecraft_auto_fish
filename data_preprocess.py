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
import getpass
from PIL import Image
from utils import image_cutter, image_cutter_with_random_shifting

user = getpass.getuser()

raw_path = "/Users/{}/Nextcloud/Documents/MineCraft/2020-02-26/raw_dataset".format(user)
dest_path = "/Users/{}/Nextcloud/Documents/MineCraft/2020-02-26/dataset2".format(user)

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
            
            # cut image with raw ratio
            dest = os.path.join(root.replace(raw_path, dest_path), name)
            img = Image.open(raw)
            cut_img = image_cutter(img, area_ratio)
            cut_img.save(dest)
            print("{} -> {}".format(raw, dest))
            
            # expend data with random shift
            expend_times = 2
            for i in range(expend_times):
                new_name = name.replace(".jpg", "_shift_{}.jpg".format(i+1))
                dest = os.path.join(root.replace(raw_path, dest_path), new_name)
                cut_img = image_cutter_with_random_shifting(img, area_ratio)
                cut_img.save(dest)
                print("{} -> {}".format(raw, dest))


walk_path(raw_path)
