import cv2
import face_recognition
import numpy as np

imgelon = face_recognition.load_image_file('face reco project assets/elonmusk.jpeg')
imgelon = cv2.cvtColor(imgelon, cv2.COLOR_BGR2RGB)
imgtest = face_recognition.load_image_file('face reco project assets/elontest.jpeg')
imgtest = cv2.cvtColor(imgtest, cv2.COLOR_BGR2RGB)

faceloc = face_recognition.face_locations(imgelon)[0]
encodeelon = face_recognition.face_encodings(imgelon)[0]
cv2.rectangle(imgelon, (faceloc[3], faceloc[0]), (faceloc[1], faceloc[2]), (255, 0, 255), 2)
faceloctest = face_recognition.face_locations(imgtest)[0]
encodetest = face_recognition.face_encodings(imgtest)[0]
cv2.rectangle(imgtest, (faceloctest[3], faceloctest[0]), (faceloctest[1], faceloctest[2]), (255, 0, 255), 2)

results = face_recognition.compare_faces([encodeelon], encodetest)
facedist = face_recognition.face_distance([encodeelon], encodetest)
print(results, facedist)

# Fixed line below
cv2.putText(imgtest, f'Results: {results} Distance: {facedist[0]:.2f}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))

cv2.imshow('elon musk', imgelon)
cv2.imshow('elon test', imgtest)
cv2.waitKey(0)
