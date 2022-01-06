import cv2
from face_recognition.api import face_locations
import numpy as np
import face_recognition

imgAyse=face_recognition.load_image_file('images/ayse_celik.jpg')
imgAyse=cv2.cvtColor(imgAyse,cv2.COLOR_BGR2RGB)
imgTest=face_recognition.load_image_file('images/ayse_celik_test.jpg')
imgTest=cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)
#deneme yaptık sonuç false çıkıyor
#imgTest2=face_recognition.load_image_file('images/1.jpg')
#imgTest2=cv2.cvtColor(imgTest2,cv2.COLOR_BGR2RGB)

faceloc=face_recognition.face_locations(imgAyse)[0]
encodeAyse=face_recognition.face_encodings(imgAyse)[0]
cv2.rectangle(imgAyse,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)

facelocTest=face_recognition.face_locations(imgTest)[0]
encodeAyseTEst=face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(facelocTest[3],facelocTest[0]),(facelocTest[1],facelocTest[2]),(255,0,255),2)
#deneme yapmak için kullanıdk
#facelocTest2=face_recognition.face_locations(imgTest2)[0]
#encodeAyseTEst2=face_recognition.face_encodings(imgTest2)[0]
#cv2.rectangle(imgTest2,(facelocTest2[3],facelocTest2[0]),(facelocTest2[1],facelocTest2[2]),(255,0,255),2)

results=face_recognition.compare_faces([encodeAyse],encodeAyseTEst)
facedis=face_recognition.face_distance([encodeAyse],encodeAyseTEst)
print(results,facedis)
cv2.putText(imgTest,f'{results} {round(facedis[0],2)}', (50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

#results2=face_recognition.compare_faces([encodeAyse],encodeAyseTEst2)
#facedis2=face_recognition.face_distance([encodeAyse],encodeAyseTEst2)
#print(results2,facedis2 )
#cv2.putText(imgTest2,f'{results2} {round(facedis2[0],2)}', (50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)


cv2.imshow('Ayse Celik',imgAyse)
cv2.imshow('Ayse Test',imgTest)
#denem için
# cv2.imshow('Ayse Test2',imgTest2)


cv2.waitKey(0)