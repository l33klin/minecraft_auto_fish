#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@Time    : 2020-02-25 11:45
@Author  : Jann
@Contact : l33klin@foxmail.com
@Site    : 
@File    : main.py
"""
import getpass
import argparse
import matplotlib.pyplot as plt
from PIL import Image
from utils import image_cutter, snapshot_from_video, get_video_info, capture_specific_time_frames_from_video, \
    get_video_frame_num, play_video, image_cutter_with_random_shifting

area_ratio = {
    "left": 13 / 32,
    "right": 21 / 36,
    "top": 4 / 20,
    "bottom": 19 / 32
}


def cut_test(img_path):
    raw_img = Image.open(img_path)
    cut_img = image_cutter(raw_img, area_ratio)
    w, h = cut_img.size
    print("Cut image size: width={}, height={}".format(w, h))
    cut_img.show()
    
    raw_img = Image.open("no_bite.jpg")
    cut_img = image_cutter(raw_img, area_ratio)
    w, h = cut_img.size
    print("Cut image size: width={}, height={}".format(w, h))
    cut_img.show()
    

def cut_with_random_shifting(img_path):
    raw_img = Image.open(img_path)
    cut_img = image_cutter_with_random_shifting(raw_img, area_ratio)
    w, h = cut_img.size
    print("Cut image size: width={}, height={}".format(w, h))
    cut_img.show()


def rotate_test(img_path):
    raw_img = Image.open(img_path)
    rotate_img = raw_img.transpose(Image.ROTATE_270)
    rotate_img.show()


def play_video_and_predict(video_path):
    play_video(video_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test some feature.')
    
    parser.add_argument("-v", "--video-path", dest="video_path",
                        help="path of video to play and predict.")

    parser.add_argument("-i", "--image-path", dest="image_path",
                        help="path of image to process.")

    args = parser.parse_args()
    
    if args.video_path:
        get_video_info(args.video_path)
        play_video_and_predict(args.video_path)

    if args.image_path:
        cut_with_random_shifting(args.image_path)
