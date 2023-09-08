# Visao_computacional
#Utilizando o Classificador OpenCV
#Haarcascade simples
#Passei uma grande dificuldade pelo fato do OpenCV não ler o pathing porcausa de um acento em "visÃo"
#Passei um bom tempo tbm com o fato de antes do pathing ter que colocar a letra "r" pra conseguir ler
#Primeiramente a gente cria uma string que recebe a imagem
#Segundamente adicionamos o detector haarcascade em outra string
#Terceiramente, recomenda-se que transformemos a imagem em cinza, pois funciona melhor assim.
#Portanto criamos uma string chamada imagem_cinza que recebe o comando cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
#Em seguida fazemos uma variável que determina quais condições damos ao detector, nesse caso chamamos de "faces"
#O programa recebe a imagem_cinza 
#E esses são alguns dos parâmetros: scaleFactor=1.2, minNeighbors=0,minSize=(0,0), maxSize=(0,0)

import cv2


imagem = cv2.imread(r'C:\Users\PC\Desktop\douglas\Visao-Computacional-Guia-Completo\Images\people1.jpg')

detector_facial = cv2.CascadeClassifier(r'C:\Users\PC\Desktop\douglas\Visao-Computacional-Guia-Completo\Cascades\haarcascade_frontalface_default.xml')

imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

faces = detector_facial.detectMultiScale(imagem_cinza, scaleFactor=1.2, minSize=(103,103))
print(faces)
#Pode usar as letras em inglês tbm = x,y,w,h
for(x, y, l, a) in faces:
    cv2.rectangle(imagem, (x,y), (x + l, y + a),(0,255,0),2)
cv2.imshow('faces', imagem)
cv2.waitKey(0)
