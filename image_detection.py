import json
import os

import cv2
import dlib
import numpy as np

detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

imagePath = 'my_faces/'
data = np.zeros((1, 128))  # Because compute_face_descriptor always return a 128D face descriptor, I hope you know that.
label = []

for file in os.listdir(imagePath):
    if '.jpg' in file or '.png' in file:
        fileName = file
        labelName = file.split('_')[0]  # get label
        print('current image: ', file)
        print('current label: ', labelName)

        img = cv2.imread(imagePath + file)
        if img.shape[0] * img.shape[1] > 500000:  # if image too large, sampling.
            img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        dets = detector(img, 1)  # detect face
        for k, d in enumerate(dets):
            rec = dlib.rectangle(d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom())
            shape = sp(img, rec)  # get landmark
            face_descriptor = facerec.compute_face_descriptor(img,
                                                              shape)  # always return a 128D face descriptor by resNet
            faceArray = np.array(face_descriptor).reshape((1, 128))
            data = np.concatenate((data, faceArray))
            label.append(labelName)
            cv2.rectangle(img, (rec.left(), rec.top()), (rec.right(), rec.bottom()), (0, 255, 0),
                          2)  # show face area by rectangle
            cv2.waitKey(2)
            cv2.imshow('image', img)

data = data[1:, :]  # remove the first useless data
np.savetxt('face_data.txt', data, fmt='%f')

labelFile = open('label.txt', 'w')
json.dump(label, labelFile)
labelFile.close()

cv2.destroyAllWindows()
