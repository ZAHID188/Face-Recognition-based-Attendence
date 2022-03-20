from importlib.resources import path
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime


path='images'
images=[]
PersonName=[]
MyList=os.listdir(path)
# print(MyList)
for var in MyList:
    nm=var.split('.')[0]
    current_img=cv2.imread(f'{path}/{var}')
    images.append(current_img)
    PersonName.append(nm)
print(PersonName)

def faceEncodings(images):
    encodeList=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


#print(faceEncodings(images)) #hog transformation algorithm
encodeListKnown=faceEncodings(images)
print("all encoding complted")



cap=cv2.VideoCapture(0) #laptop camera =0 external camera =1

while True:
    ret,frame =cap.read()
    faces=cv2.resize(frame,(0,0),None,0.25,0.25)
    faces=cv2.cvtColor(faces,cv2.COLOR_BGR2RGB)
    facesCurrntFrame=face_recognition.face_locations(faces)
    encodeCurrentFrame=face_recognition.face_encodings(faces,facesCurrntFrame)
    for encodeFace,faceloc in zip(encodeCurrentFrame,facesCurrntFrame):
        matches=face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis= face_recognition.face_distance(encodeListKnown,encodeFace)
        matchIndex=np.argmin(faceDis)

        if matches[matchIndex]:
            name= PersonName[matchIndex].upper()
            # print(name)
            y1,x2,y2,x1=faceloc
            y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(frame,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(frame,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
    cv2.imshow("Camera",frame)
    if cv2.waitKey(10)==13:
        break

cap.release()
cv2.destroyAllWindows()
