import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime

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

def markAttendance(name, action):
    with open('attendanceimg/attendance.csv','a') as f:
        now = datetime.now()
        dtString = now.strftime('%H:%M:%S')
        f.write(f'\n{name},{dtString},{action}')

encodelistknown = findEncodings(images)
print(len(encodelistknown))

# Initialize webcam
cap = cv2.VideoCapture(0)

# Check if webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
else:
    while True:
        command = input("Please type 'entry' to mark entry or 'exit' to mark exit: ")

        if command.lower() == 'entry':
            while True:
                success, img = cap.read()
                if not success:
                    print("Error: Failed to capture frame from webcam.")
                    break
                
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
                        markAttendance(name, "Entry")

                cv2.imshow('webcam',img)
                if cv2.waitKey(1) == ord('q'):
                    break

        elif command.lower() == 'exit':
            # Reopen webcam
            cap.release()
            cap = cv2.VideoCapture(0)

            while True:
                success, img = cap.read()
                if not success:
                    print("Error: Failed to capture frame from webcam.")
                    break
                
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
                        markAttendance(name, "Exit")

                cv2.imshow('webcam',img)
                if cv2.waitKey(1) == ord('q'):
                    break

        else:
            print("Invalid command. Please type 'entry' or 'exit'.")

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
