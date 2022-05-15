import cv2 as cv
import os
import numpy as np

people=['Ben Afflek','Elton John','Jerry Seinfield','Madonna','Mindy Kaling']
haar=cv.CascadeClassifier('harr.xml')
dir=r'D:\pythonProject\face_detect\opencv-course-master\opencv-course-master\Resources\Faces\train'

features=[]
labels=[]
def c_train():

    for p in people:
        dir1=os.path.join(dir,p)
        l=people.index(p)
        for img in os.listdir(dir1):
            img_path=os.path.join(dir1,img)
            i_arr=cv.imread(img_path)
            gray=cv.cvtColor(i_arr,cv.COLOR_BGR2GRAY)
            face_rec=haar.detectMultiScale(gray,1.1,3)
            for (x,y,w,h) in face_rec:
                img1=gray[y:y+h,x:x+w]
                f1=features.append(img1)
                l1=labels.append(l)
c_train()
f4=np.array(f1, dtype='object')
l4=np.array(l1)

f5=cv.face.LBPHFaceRecognizer_create()
f5.train(f4,l4)

f5.save('f10.yml')
np.save('feat.npy',f4)
np.save('lab.npy',l4)
