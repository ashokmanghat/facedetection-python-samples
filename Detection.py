import numpy as np
import cv2


face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_eye.xml')
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read("trainingdata.yml")
cap = cv2.VideoCapture(0)
id=0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for (x,y,w,h) in faces:
       frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
       id,conf=rec.predict(gray[y:y+h,x:x+w])
       if(id==1):
           id='Ashok'     
       if(id==2):
           id='Aji'   
       roi_gray = gray[y:y+h, x:x+w]
       roi_color = frame[y:y+h, x:x+w]
       eyes = eye_cascade.detectMultiScale(roi_gray)
       cv2.putText(frame,str(id),(x-2,y), font, 0.4, (0,255,0), 1, cv2.LINE_AA)
       for (ex,ey,ew,eh) in eyes:
           cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
           
       
   
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
# img = cv2.imread('sachin.jpg')

cv2.imshow('Detection',frame)
cv2.waitKey(0)
cv2.destroyAllWindows()