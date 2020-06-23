#importing the Libraries

import numpy as np
import cv2
import os
import covidmask as cm

mask={0:'Masked',
    1:'Put your Mask on'
  }



# Init Camera
cap = cv2.VideoCapture(0)
# Face Detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
while True:
    ret,frame = cap.read()
    # frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    if ret == False:
        continue
    faces = face_cascade.detectMultiScale(frame,1.3,5)

    for face in faces:
        x,y,w,h = face

        #Extract (Crop out the required Area) : region Of interest
        offset =10
        face_section = frame[y-offset:y+offset+h,x-offset:x+offset+w]
        face_section = cv2.resize(face_section,(150,150))
        face_section = (face_section.reshape(1,150,150,3))/255.0
        out = cm.model.predict(face_section)
        print(out)
        out = out[0][0]>0.5

        pred_name = mask[out]
        cv2.putText(frame,pred_name,(x,y+h+25),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

    cv2.imshow("FACES",frame)
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
