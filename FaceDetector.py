import cv2
import sys
from time import sleep
import numpy as np

cascPath = "haarcascade_frontalface_default.xml"
cascPath1 = "haarcascade_eye.xml"

eyeCascade = cv2.CascadeClassifier(cascPath1)
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)
anterior = 0
while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    eyes = eyeCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    if anterior != len(faces):
        anterior = len(faces)

        font                   = cv2.FONT_HERSHEY_COMPLEX
        bottomLeftCornerOfText = (10,500)
        fontScale              = 2
        fontColor              = (255,255,255)
        lineType               = 2

        cv2.putText(frame,str(len(faces)),
            bottomLeftCornerOfText,
            font,
            fontScale,
            fontColor,
            lineType)


        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        
        cv2.imshow('Video', frame)
video_capture.release()
cv2.destroyAllWindows()
