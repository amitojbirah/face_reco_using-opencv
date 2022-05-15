import cv2 as cv
import numpy as np
import os

people=['Ben Afflek','Elton John','Jerry Seinfield','Madonna','Mindy Kaling']

haar=cv.CascadeClassifier('harr.xml')
fr=cv.face.LBPHFaceRecognizer_create()
fr.read('f10.yml')

img=cv.imread('sdadnsj')
gr=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
h1=haar.detectMultiScale(gr,1.1,3)

for (x,y,w,h) in h1:
    y1=gr[y:y+h,x:x+w]

    y2,conf=fr.predict(y1)
    print(f'label={people[label]} with confidence of {confidence}')
    cv.putText(img, str(people[label]), (20, 20), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 1)
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv.imshow("img", img)
cv.waitKey(0)