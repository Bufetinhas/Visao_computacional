# Visao_computacional
 Esse é um projeto que usa IA para detectar rostos, olhos, carros etc.
#Esse código visa detectar faces e olhos ao mesmo tempo, 

import cv2


imagem = cv2.imread(r'C:\Users\PC\Desktop\douglas\Visao-Computacional-Guia-Completo\Images\people1.jpg')

#imagem = cv2.resize(imagem,(800, 600)), muito util para detectar faces, mas atrapalha ao detectar olhos, porcausa da resolução da imagem.

detector_facial = cv2.CascadeClassifier(r'C:\Users\PC\Desktop\douglas\Visao-Computacional-Guia-Completo\Cascades\haarcascade_frontalface_default.xml')

detector_olhos = cv2.CascadeClassifier(r'C:\Users\PC\Desktop\douglas\Visao-Computacional-Guia-Completo\Cascades\haarcascade_eye.xml')

imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

deteccoes = detector_facial.detectMultiScale(imagem_cinza, scaleFactor=1.3, minSize=(30,30))
#print(deteccoes)

for(x, y, l, a) in deteccoes:
    cv2.rectangle(imagem, (x,y), (x + l, y + a),(0,255,0),2)

deteccoes_olhos = detector_olhos.detectMultiScale(imagem_cinza, scaleFactor=1.09, minNeighbors=10, maxSize=(70,70))
for(x, y, l, a) in deteccoes_olhos:
    cv2.rectangle(imagem, (x,y), (x + l, y + a),(0,255,0),2)

cv2.imshow('deteccoes', imagem)
cv2.waitKey(0)