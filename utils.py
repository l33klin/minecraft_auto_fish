#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@Time    : 2020-02-25 11:54
@Author  : Jann
@Contact : l33klin@foxmail.com
@Site    : 
@File    : utils.py
"""
import cv2
import os
import time
import base64
import queue
import requests
import threading
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from collections import namedtuple

area_ratio = {
    "left": 13 / 32,
    "right": 21 / 36,
    "top": 4 / 20,
    "bottom": 19 / 32
}

# for plot
N = 100
xq = queue.Queue(N)
plot_queue = queue.Queue(N)
# x = [i for i in range(N)]
for i in range(N):
    xq.put(i)
    plot_queue.put(0.0)

yaxis_range = (-5, 5)
# init figure
# plt.axis([0, 100, -10, 10])
plt.ion()
plt.show()


def get_video_info(video_path):
    """
    Refs: https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-get
    :param video_path:
    :return:
    """
    # Start default camera
    video = cv2.VideoCapture(video_path)
    
    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    
    # With webcam get(CV_CAP_PROP_FPS) does not work.
    # Let's see for ourselves.
    
    if int(major_ver) < 3:
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else:
        fps = video.get(cv2.CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
    
    total_frame = video.get(cv2.CAP_PROP_FRAME_COUNT)
    print("Total frames: {}".format(total_frame))
    
    video.set(cv2.CAP_PROP_POS_FRAMES, total_frame)
    msec = video.get(cv2.CAP_PROP_POS_MSEC)
    print("MSEC: {}".format(msec))
    
    print("Calculate sec: {}".format(total_frame/fps))
    
    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print("video frame size: {} * {}".format(width, height))
    
    # Release video
    video.release()


def snapshot_from_video(video_path, dest_path, snapshot_interval=None):
    """

    Ref: https://stackoverflow.com/questions/57791203/python-take-screenshot-from-video
    https://www.geeksforgeeks.org/extract-images-from-video-in-python/
    https://stackoverflow.com/questions/33311153/python-extracting-and-saving-video-frames
    https://www.learnopencv.com/how-to-find-frame-rate-or-frames-per-second-fps-in-opencv-python-cpp/
    """
    video_name = os.path.split(video_path)[-1].split('.')[0]
    print("video name: ", video_name)
    print("Press Enter to continue...")
    cv2.waitkey()

    # Read the video from specified path
    cam = cv2.VideoCapture(video_path)
    fps = cam.get(cv2.CAP_PROP_FPS)  # get FPS
    
    total_frame = cam.get(cv2.CAP_PROP_FRAME_COUNT)
    print("Total frames: {}".format(total_frame))
    
    try:
        # creating a folder named data
        if not os.path.exists('data'):
            os.makedirs('data')

            # if not created then raise error
    except OSError:
        print('Error: Creating directory of data')

    # frame
    current_frame = 0
    frame_step = int(snapshot_interval * fps)

    while True:

        # reading from frame
        ret, frame = cam.read()

        if ret:
            # if video is still left continue creating images
            name = os.path.join(dest_path, '{}_frame_{}.jpg'.format(video_name, current_frame))
            print('Creating... ' + name)

            # writing the extracted images
            cv2.imwrite(name, frame)
            
            print("Current pos msec: {}".format(cam.get(cv2.CAP_PROP_POS_MSEC)))
            current_frame += frame_step
            cam.set(cv2.CAP_PROP_POS_FRAMES, current_frame)  # set next video capture timestamp
        else:
            break

    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()


def capture_specific_time_frames_from_video(video_path, time_series, dest_path, extend_time=2, rotate_angle=0):
    """
    不好用，越到后面误差越大
    :param video_path: path of video
    :param time_series: 截取的时间点序列
    :param extend_time: 前后扩展的时间，如截取时间点是23秒，扩展时间为2秒，则截取21-25秒这四秒的内容
    :param rotate_angle: 画面旋转角度，如果需要顺时针旋转90度才能使图像正过来，则输入90
    :return:
    """
    video_name = os.path.split(video_path)[-1].split('.')[0]
    print("video name: ", video_name)

    cam = cv2.VideoCapture(video_path)
    total_frame = cam.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cam.get(cv2.CAP_PROP_FPS)
    print("Total frames: {}".format(total_frame))
    print("FPS: {}".format(fps))
    total_secs = total_frame/fps
    print("Total seconds: {}".format(total_secs))
    input("Press Enter to continue...")
    
    Frames = namedtuple("Frames", ["start", "end"])
    frame_slices = []
    for time_ in time_series:
        time_frame = int(time_ * fps)
        frame_slices.append(Frames(int(time_frame - extend_time * fps), int(time_frame + extend_time * fps)))
    print("Frame slices: {}".format(frame_slices))
    input("Press Enter to continue...")

    index = 0           # index to traverse all frame slice
    frame_step = 6
    actual_frames = 0
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        actual_frames += 1
        current_frame_num = int(cam.get(cv2.CAP_PROP_POS_FRAMES))
        
        if current_frame_num > frame_slices[index].end:
            index += 1
            if index >= len(frame_slices):
                break

        if current_frame_num > frame_slices[index].start and current_frame_num % frame_step == 0:
            # if video is still left continue creating images
            name = os.path.join(dest_path, '{}_frame_{}.jpg'.format(video_name, current_frame_num))
            print('Creating... ' + name)

            # convert cv2 image to PIL image
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # rotate picture if need
            if rotate_angle != 0:
                # frame = rotate_img_by_using_cv2(frame, rotate_angle)
                image = rotate_image(image, rotate_angle)
            # writing the extracted images
            image.save(name)
    print("The end frame number: ", actual_frames)
    cam.release()
    cv2.destroyAllWindows()


def capture_specific_frames_from_video(video_path, frame_series, dest_path, extend_time=2, rotate_angle=0):
    """
    :param video_path: path of video
    :param frame_series: 截取的帧序列
    :param extend_time: 前后扩展的时间，如截取时间点是23秒，扩展时间为2秒，则截取21-25秒这四秒的内容
    :param rotate_angle: 画面旋转角度，如果需要顺时针旋转90度才能使图像正过来，则输入90
    :return:
    """
    video_name = os.path.split(video_path)[-1].split('.')[0]
    print("video name: ", video_name)
    
    cam = cv2.VideoCapture(video_path)
    total_frame = cam.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cam.get(cv2.CAP_PROP_FPS)
    print("Total frames: {}".format(total_frame))
    print("FPS: {}".format(fps))
    total_secs = total_frame / fps
    print("Total seconds: {}".format(total_secs))
    input("Press Enter to continue...")
    
    Frames = namedtuple("Frames", ["start", "end"])
    frame_slices = []
    for specific_frame in frame_series:
        frame_slices.append(Frames(int(specific_frame - extend_time * fps), int(specific_frame + extend_time * fps)))
    print("Frame slices: {}".format(frame_slices))
    input("Press Enter to continue...")
    
    index = 0  # index to traverse all frame slice
    frame_step = 6
    actual_frames = 0
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        actual_frames += 1
        current_frame_num = int(cam.get(cv2.CAP_PROP_POS_FRAMES))
        
        if current_frame_num > frame_slices[index].end:
            index += 1
            if index >= len(frame_slices):
                break
        
        if current_frame_num > frame_slices[index].start and current_frame_num % frame_step == 0:
            # if video is still left continue creating images
            name = os.path.join(dest_path, '{}_frame_{}.jpg'.format(video_name, current_frame_num))
            print('Creating... ' + name)
            
            # convert cv2 image to PIL image
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # rotate picture if need
            if rotate_angle != 0:
                # frame = rotate_img_by_using_cv2(frame, rotate_angle)
                image = rotate_image(image, rotate_angle)
            # writing the extracted images
            image.save(name)
    print("The end frame number: ", actual_frames)
    cam.release()
    cv2.destroyAllWindows()
    

def rotate_img_by_using_cv2(cv2_img, angle):
    """
    WARNING: ONLY WORK WHEN IMAGE IS square !!!
    
    Perform the counter clockwise rotation holding at the center by using cv2
    :param cv2_img:
    :param angle:
    :return:
    """
    # get image height, width
    (h, w) = cv2_img.shape[:2]
    # calculate the center of the image
    center = (w / 2, h / 2)
    
    scale = 1.0
    
    # Perform the counter clockwise rotation holding at the center
    # 90 degrees
    M = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(cv2_img, M, (h, w))


def get_video_frame_num(video_path):
    """
    :param

    Ref: https://stackoverflow.com/questions/57791203/python-take-screenshot-from-video
    https://www.geeksforgeeks.org/extract-images-from-video-in-python/
    https://stackoverflow.com/questions/33311153/python-extracting-and-saving-video-frames
    https://www.learnopencv.com/how-to-find-frame-rate-or-frames-per-second-fps-in-opencv-python-cpp/
    """
    video_name = os.path.split(video_path)[-1].split('.')[0]
    print("video name: ", video_name)
    input("Press Enter to continue...")
    
    # Read the video from specified path
    cam = cv2.VideoCapture(video_path)
    fps = cam.get(cv2.CAP_PROP_FPS)  # get FPS
    
    total_frame = cam.get(cv2.CAP_PROP_FRAME_COUNT)
    print("Total frames by using cam.get(cv2.CAP_PROP_FRAME_COUNT): {}".format(total_frame))
    
    # frame
    frame_sum = 0
    
    while True:
        # reading from frame
        ret, frame = cam.read()
        if ret:
            current_frame = cam.get(cv2.CAP_PROP_POS_FRAMES)
            if int(current_frame) % 1000 == 0:
                print("current frame: ", current_frame)
            frame_sum += 1
        else:
            break
    
    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()
    print("Frame sum by read all frames to count: ", frame_sum)


def play_video(video_path):
    """
    NOTE: should install opencv extras library by using: `pip install opencv-contrib-python`
    Ref: https://www.learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/
    :param video_path:
    :return:
    """
    # Create a VideoCapture object and read from input file
    cap = cv2.VideoCapture(video_path)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error opening video file")
        
    print("Ready...")
    while cap.isOpened():
        # time.sleep(0.01)
        # Capture frame-by-frame
        ret, frame = cap.read()
        x = None
        if ret:
            current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            
            if int(current_frame) % 6 == 0:
                if x:
                    x.join()
                update_figure()
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                # image = rotate_image(image, 90)
                image = image_cutter(image, area_ratio)
                buffered = BytesIO()
                image.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue())
                x = threading.Thread(target=post_to_predict, args=(img_str,))
                x.start()
            
            # Display the resulting frame
            cv2.imshow('Frame', frame)
            # print("Current frame: ", current_frame)
            
            # # Press Q on keyboard to  exit
            # if cv2.waitKey(25) & 0xFF == ord('q'):
            #     break
            #
            # # Press Q on keyboard to  exit
            # if cv2.waitKey(25) & 0xFF == ord('p'):
            #     print("Target Frame: {}".center(50, '='))
        
        # Break the loop
        else:
            break
    
    # When everything done, release
    # the video capture object
    cap.release()
    
    # Closes all the frames
    cv2.destroyAllWindows()


def rotate_image(img, angle):
    if angle == 90:
        return img.transpose(Image.ROTATE_90)
    elif angle == 180:
        return img.transpose(Image.ROTATE_180)
    elif angle == 270:
        return img.transpose(Image.ROTATE_270)
    else:
        raise Exception("angle must in [90, 180, 270]")


def image_cutter(img, area):
    # im = Image.open(img_path)

    # Size of the image in pixels (size of orginal image)
    # (This is not mandatory)
    width, height = img.size

    # Setting the points for cropped image
    left = width * area['left']
    top = height * area['top']
    right = width * area['right']
    bottom = height * area['bottom']

    # Cropped image of above dimension
    # (It will not change orginal image)
    cut_img = img.crop((left, top, right, bottom))

    # Shows the image in image viewer
    # cut_img.show()
    
    return cut_img


def post_to_predict(img_bytes):
    start = time.time()
    _url = 'http://127.0.0.1:8007/predict'
    body = {
        "bs64_img": img_bytes.decode('utf8')
    }
    response = requests.post(_url, json=body)
    prediction = response.json()['prediction']
    put_prediction(prediction)
    print("predict cost time: {}".format(time.time() - start))


def put_prediction(prediction):
    """
    Put prediction to queue, Can not plot in sub-thread
    :param prediction:
    :return:
    """
    try:
        xq.get(block=False)
        plot_queue.get(block=False)
    except queue.Empty:
        pass
    global N
    xq.put(N)
    N += 1
    plot_queue.put(prediction)


def update_figure():
    """
    Should call in main thread
    :return:
    """
    plt.clf()
    # plt.close()
    plt.ylim(*yaxis_range)  # set y axis range
    plt.autoscale(enable=False, axis='y', tight=True)  # disable auto scale
    plt.scatter(list(xq.queue), list(plot_queue.queue))
    plt.draw()
    plt.pause(0.001)
