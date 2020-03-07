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
import matplotlib.pyplot as plt
from PIL import Image
from utils import image_cutter, snapshot_from_video, get_video_info, capture_specific_time_frames_from_video, \
    get_video_frame_num, play_video, capture_specific_frames_from_video

area_ratio = {
    "left": 13 / 32,
    "right": 21 / 36,
    "top": 4 / 20,
    "bottom": 19 / 32
}
# raw_img = Image.open("bite.jpg")
# cut_img = image_cutter(raw_img, area_ratio)
# w, h = cut_img.size
# print("Cut image size: width={}, height={}".format(w, h))
# cut_img.show()
#
# raw_img = Image.open("no_bite.jpg")
# cut_img = image_cutter(raw_img, area_ratio)
# w, h = cut_img.size
# print("Cut image size: width={}, height={}".format(w, h))
# cut_img.show()
#
# rotate_img = raw_img.transpose(Image.ROTATE_270)
# rotate_img.show()

video_path = "/Users/klin/Downloads/Minecraft/fish_video/2020-03-07_13-19.mp4"
# video_path = "/Users/{}/Nextcloud/Documents/MineCraft/2020-02-26/2020-02-26_17-27_demo.mp4".format(getpass.getuser())
dest_path = "./data/image2/"
# "/Users/klin/Nextcloud/Documents/MineCraft/2020-02-26/unlabeled"

# get_video_info(video_path)
# snapshot_from_video(video_path, dest_path)
# get_video_info("./data/RPReplay_Final1582691113_clip1.mp4")
bite_time_series = [40.5, 52, 59, 92.5, 109, 134, 143, 165, 172, 206, 213, 231, 263, 280, 293, 317, 328, 337, 351, 357,
                    363, 383, 391, 411, 417, 423, 434, 447, 455, 490, 512, 524, 555, 575, 581, 610]
# capture_specific_time_frames_from_video(video_path, bite_time_series, dest_path, extend_time=2, rotate_angle=90)
frame_series = [2400, 3000, 3450, 5450, 6350, 7860, 8350, 9680, 10050, 12050, 12420, 13450, 15350, 16300, 17050, 18440,
                19050, 19590, 20400, 20710, 21090, 21090, 22690, 23800, 24150, 24540, 25100, 25900, 26390, 28310, 26580,
                30270, 32040, 33100, 33560, 35210, ]
# capture_specific_frames_from_video(video_path, frame_series, dest_path, extend_time=2, rotate_angle=90)
# get_video_frame_num(video_path)
play_video(video_path)
