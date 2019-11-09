import cv2

# algoritimo de treinamento de face humana
classificador = cv2.CascadeClassifier("haarcascade-frontalface-default.xml")
classificadorOlho = cv2.CascadeClassifier("haarcascade-eye.xml")

camera = cv2.VideoCapture(0)
amostra = 1
numeroAmostras = 25
id = input("Digite seu identificador: ")

largura, altura = 220, 220
print("Capturando as faces....")

while (True):
    conectado, imagem = camera.read()
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) # É MELHOR TRABALHAR COM IMG CINZA
    facesDetectadas = classificador.detectMultiScale(imagemCinza, 
                                                     scaleFactor=1.5,
                                                     minSize=(150, 150))

    # criamos o retangulo vermelho em volta da face
    for(x, y, l, a) in facesDetectadas:
        cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)

        regiao = imagem[y:y + a, x:x + l]
        regiaoCinzaOlho = cv2.cvtColor(regiao, cv2.COLOR_BGR2GRAY)  # É MELHOR TRABALHAR COM IMG CINZA
        olhosDetectados = classificadorOlho.detectMultiScale(regiaoCinzaOlho)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura, altura))
            cv2.imwrite("photos/pessoa.{0}.{1}.jpg".format(str(id), str(amostra)), imagemFace)
            print("foto {0} capturada com sucesso".format( str(amostra)))
            amostra += 1

    cv2.imshow("Face", imagem)
    cv2.waitKey(1)
    if (amostra >= numeroAmostras+1):
        break

print("Faces capturadas")
camera.release()
cv2.destroyAllWindows()

