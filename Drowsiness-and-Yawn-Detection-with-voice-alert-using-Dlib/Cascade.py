import cv2
from imutils.video import VideoStream
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0,
                help="index of webcam on system")
args = vars(ap.parse_args())

print("-> Loading the predictors and detectors...")

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


print("-> Starting Video Stream")
vs = VideoStream(src=args["webcam"]).start()

time.sleep(1.0)

#cap = cv2.VideoCapture(0)
while True:

    img = vs.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (f_x, f_y, f_w, f_h) in faces:
        cv2.rectangle(img, (f_x, f_y), (f_x + f_w, f_y + f_w), (255, 0, 0), 2)
        roi_gray = gray[f_y:f_y + f_h, f_x:f_x + f_w]
        roi_color = img[f_y:f_y + f_h, f_x:f_x + f_w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (e_x, e_y, e_w, e_h) in eyes:
            cv2.rectangle(roi_color, (e_x, e_y), (e_x + e_w, e_y + e_h), (0, 255, 0), 2)

    cv2.imshow('img',img)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
