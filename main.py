#importing modules
import cv2
import numpy as np
import face_recognition

#import the image(for test)
imgDhoni = face_recognition.load_image_file('imageAttendance/Ms Dhoni.jpg')  #function to load image
imgDhoni = cv2.cvtColor(imgDhoni,cv2.COLOR_BGR2RGB) #converting img into RGB
imgTest = face_recognition.load_image_file('imageAttendance/Rohit sharma.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

#FINDING THE FACES IN OUR IMAGE AND ENCODINGS AS WELL
faceLoc = face_recognition.face_locations(imgDhoni)[0] #SENDING 1STIMG 
encodeDhoni = face_recognition.face_encodings(imgDhoni)[0] #ENCODING IMG AS ITS 1ST ELEMENT
# TO SEE WHERE WE HAVE DETECTED THE FACE
cv2.rectangle(imgDhoni,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

#CHECKING TRUE OR FALSE IF IT MATCHES TO THE IMAGE
results = face_recognition.compare_faces([encodeDhoni],encodeTest)
faceDis = face_recognition.face_distance([encodeDhoni],encodeTest) #DISTANCE OF IMG (lower the distance the better match)
cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
print(results,faceDis)
cv2.imshow('Ms Dhoni',imgDhoni)
cv2.imshow('Ms DhoniTest',imgTest)
cv2.waitKey(0)