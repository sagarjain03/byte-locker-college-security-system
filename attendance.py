import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime
import threading

path = 'facerecoimg'
images = []
classnames = []
mylist = os.listdir(path)
print(mylist)

for cls in mylist:
    curImg = cv2.imread(f'{path}/{cls}')  
    images.append(curImg)
    classnames.append(os.path.splitext(cls)[0])
print(classnames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('attendanceimg/attendance.csv','a') as f:
        now = datetime.now()
        dtString = now.strftime('%H:%M:%S')
        f.write(f'\n{name},{dtString},Entry')

def markExit(name):
    with open('attendanceimg/attendance.csv', 'a') as f:  
        now = datetime.now()
        exit_time = now.strftime('%H:%M:%S')
        f.write(f'\n{name},{exit_time},Exit')  

encodelistknown = findEncodings(images)
print(len(encodelistknown))

def capture_loop():
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facescurframe = face_recognition.face_locations(imgS)
        encodescurframe = face_recognition.face_encodings(imgS, facescurframe)

        for encodeface, faceloc in zip(encodescurframe, facescurframe):
            matches = face_recognition.compare_faces(encodelistknown, encodeface)
            facedis = face_recognition.face_distance(encodelistknown, encodeface)
            matchindex = np.argmin(facedis)

            if matches[matchindex]:
                name = classnames[matchindex].upper()
                y1, x2, y2, x1 = faceloc
                y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
                markAttendance(name)

        cv2.imshow('webcam',img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def input_loop():
    while True:
        command = input("Type 'scan' to mark exit: ")
        if command.lower() == 'scan':
            name = input("Enter the name of the person exiting: ")
            markExit(name)
            print(f"{name} marked as exited.")


capture_thread = threading.Thread(target=capture_loop)
capture_thread.start()

input_loop()
