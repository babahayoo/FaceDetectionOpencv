import numpy as np
import cv2 as cv

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_alt2.xml')
print(face_cascade)
cap = cv.VideoCapture(0)


while(True):
    #Capture frame by frame

    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        print(x,y,w,h)
        roi_gray = gray[y:y+h , x:x+w] #y cord start, y cord end
        roi_color = frame[y:y + h, x:x + w]


        img_item = "my-image.png"
        cv.imwrite(img_item,roi_gray)

        color = (255,0,0)
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv.rectangle(frame,(x,y),(end_cord_x,end_cord_y),color,stroke)



    #display the resulting frame
    cv.imshow('Face Recognition',frame)
    if cv.waitKey(20) & 0xFF == ord('q'):
        break

#when everything done, release the capture
cap.release()

