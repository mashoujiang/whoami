# -*-coding: utf-8 -*-
# !/usr/bin/python3
"""
    collect 10000 images from camera and labeled.
"""
import os
import random

import cv2
import dlib

root_path = os.getcwd()
output_dir = root_path + '/my_faces'
file_prefix = 'murphy_'  # this is a label we used for detection later, please use correct name label instead.
size = 64

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# adjust brightness and contrast
def relight(img, light=1, bias=0):
    w = img.shape[1]
    h = img.shape[0]
    for i in range(0, w):
        for j in range(0, h):
            for c in range(3):
                tmp = int(img[j, i, c] * light + bias)
                if tmp > 255:
                    tmp = 255
                elif tmp < 0:
                    tmp = 0
                img[j, i, c] = tmp
    return img


# Extract features by dlib frontal_face_detector
detector = dlib.get_frontal_face_detector()
# Open camera, para=video stream, you can use local video file instead '0', e.g. 'video.mp4'.
camera = cv2.VideoCapture(0)

index = 1
while True:
    if index <= 10000:
        print('Being processed picture %s' % index)
        # get image from camera
        success, img = camera.read()
        # RBG --> GRAY
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # The 1 in the second argument indicates that we should upsample the image
        # 1 time.  This will make everything bigger and allow us to detect more
        # faces.
        dets = detector(gray_img, 1)

        for i, d in enumerate(dets):
            x1 = d.top() if d.top() > 0 else 0
            y1 = d.bottom() if d.bottom() > 0 else 0
            x2 = d.left() if d.left() > 0 else 0
            y2 = d.right() if d.right() > 0 else 0

            face = img[x1:y1, x2:y2]
            # Increasing diversity in the samples
            face = relight(face, random.uniform(0.5, 1.5), random.randint(-50, 50))
            face = cv2.resize(face, (size, size))
            cv2.imshow('image', face)
            cv2.imwrite(output_dir + '/' + file_prefix + str(index) + '.jpg', face)

            index += 1
        key = cv2.waitKey(30) & 0xff
        # Exit when get 'ESC'
        if key == 27:
            break
    else:
        print('Finished!')
        break
