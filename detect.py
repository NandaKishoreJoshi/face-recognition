import cv2
import numpy as np
import face_recognition
import os

path='images'
images=[]
classNames=[]
mylist=os.listdir(path)
print((mylist))

for cl in mylist:
    curimg=cv2.imread(f'{path}/{cl}')
    images.append(curimg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findencoding(images):
    encodeList=[]
    for image in images:
        img=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        imgencode=face_recognition.face_encodings(img)[0]
        encodeList.append(imgencode)
    return encodeList

encodeListKnown=findencoding(images)
print('encoding complete')

cap=cv2.VideoCapture(0)

while True:
    success, img=cap.read()
    imgs=cv2.resize(img,(0,0),None,0.25,0.25)
    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)

    faceLocCurFrame = face_recognition.face_locations(imgs)
    encodeCurFrame = face_recognition.face_encodings(imgs,faceLocCurFrame)

    for encodeFace,faceLoc in zip(encodeCurFrame,faceLocCurFrame):
        matches=face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis=face_recognition.face_distance(encodeListKnown,encodeFace)
        print('face distnace',faceDis)
        print(matches)
        matchIndex=np.argmin(faceDis)

        if matches[matchIndex] & (faceDis[matchIndex] < 0.45):
            name=classNames[matchIndex].upper()

            y1,x2,y2,x1=faceLoc
            y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)

    cv2.imshow('webcam',img)
    cv2.waitKey(1)





