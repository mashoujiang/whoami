import json

import cv2
import dlib
import numpy as np

detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
threshold = 0.54


def find_nearest_class_for_image(face_descriptor, face_label):
    temp = face_descriptor - data
    temp = np.reshape(temp, [-1, 128])
    e = np.linalg.norm(temp, axis=1, keepdims=True)
    min_distance = e.min()
    # print('distance: ', min_distance)
    if min_distance > threshold:
        return 'other'
    index = np.argmin(e)
    return face_label[index]


def recognition(img):
    dets = detector(img, 1)
    for k, d in enumerate(dets):
        # print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        #     k, d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom()))
        rec = dlib.rectangle(d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom())
        # print(rec.left(), rec.top(), rec.right(), rec.bottom())
        shape = sp(img, rec)
        face_descriptor = facerec.compute_face_descriptor(img, shape)

        class_pre = find_nearest_class_for_image(face_descriptor, label)
        # print(class_pre)
        cv2.rectangle(img, (rec.left(), rec.top() + 10), (rec.right(), rec.bottom()), (0, 255, 0), 2)
        cv2.putText(img, class_pre, (rec.left(), rec.top()), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('image', img)


labelFile = open('label.txt', 'r')
label = json.load(labelFile)  # load local label
labelFile.close()

data = np.loadtxt('face_data.txt', dtype=float)  # load local feature
video_file = "video.mp4"
cap = cv2.VideoCapture(video_file)

while True:
    _, frame = cap.read()
    # frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
    if frame is not None:
        recognition(frame)
    else:
        break
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
