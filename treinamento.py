import cv2
import os
import numpy as np

eigenface = cv2.face.EigenFaceRecognizer_create()
fisherface = cv2.face.FisherFaceRecognizer_create()
lbph = cv2.face.LBPHFaceRecognizer_create()

def getImagemComId():
    caminhos = [os.path.join('photos', f) for f in os.listdir('photos')]
    faces = []
    ids = []
    for caminhoImagem in caminhos:
        imagemFace = cv2.cvtColor(cv2.imread(caminhoImagem), cv2.COLOR_BGR2GRAY)
        id = int(os.path.split(caminhoImagem)[-1].split('.')[1])
        ids.append(id)
        faces.append(imagemFace)

    return np.array(ids), faces

ids, faces = getImagemComId()

print("Treinando....")

#aprendizado supervisionado
eigenface.train(faces, ids) # aqui vai fazer o treinamento

eigenface.write("classificadorEigen.yml") # vamos usar esse arquivo posteriormente para fazer o reconhecimento facil

fisherface.train(faces, ids)
fisherface.write('classificadorFisher.yml')

lbph.train(faces, ids)
lbph.write('classificadorLBPH.yml')

print("Treinamento finalizado...")