import cv2
import mediapipe as mp
import math
import numpy as np
import random
import time
from tqdm import tqdm

# RANDOM
def emocionAleatoria():
    emocion = {
        1: 'Enojo',
        2: 'Felicidad',
        3: 'Asombro',
        4: 'Tristeza',
        5: 'Neutralidad'
    }
    return emocion.get(random.randint(1, 5), 'Neutralidad')

# EMOJIS
def imagenEmocion(emocion):
    if emocion == 'Enojo':
        imagen = cv2.imread('enojo.jpeg')
    elif emocion == 'Felicidad':
        imagen = cv2.imread('felicidad.jpeg')
    elif emocion == 'Asombro':
        imagen = cv2.imread('sorpresa.jpeg')
    elif emocion == 'Tristeza':
        imagen = cv2.imread('tristeza.jpeg')
    else: # Neutralidad
        imagen = cv2.imread('neutralidad.jpeg')
    return imagen

cap = cv2.VideoCapture(0)

mpDibujo = mp.solutions.drawing_utils
ConfDibu = mpDibujo.DrawingSpec(thickness=1, circle_radius=1)

mpMallaFacial = mp.solutions.face_mesh
MallaFacial = mpMallaFacial.FaceMesh(max_num_faces=1)

limite = 7 # Puntuación para terminar
emocion = emocionAleatoria()
anterior = emocion # Para que no haya repetición seguida
contador = 0
puntuacion = 0
loop = tqdm(total=limite, position=0, leave=False) # Progreso

while True:
    ret, frame = cap.read()
    nFrame = cv2.hconcat([frame, np.zeros((480, 300, 3), np.uint8)])
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Observanos los resultados
    resultados = MallaFacial.process(frameRGB)
    # Creamos unas listas donde almacenarenos los resultados
    px = []
    py = []
    lista = []

    if resultados.multi_face_landmarks:
        for rostros in resultados.multi_face_landmarks:
            # OPCIONAL:
            # mpDibujo.draw_landmarks(frame, rostros, mpMallaFacial.FACE_CONNECTIONS, ConfDibu, ConfDibu)
            for id, puntos in enumerate(rostros.landmark):
                al, an, c = frame.shape
                x, y = int(puntos.x * an), int(puntos.y * al)
                px.append(x)
                py.append(y)
                lista.append([id, x, y])

                if len(lista) == 468:
                    # Ceja Derecha
                    x1, y1 = lista[65][1:]
                    x2, y2 = lista[158][1:]
                    cx, cy = (x1 + x2) // 2, (y1 - y2) // 2
                    longitud1 = math.hypot(x2 - x1, y2 - y1)

                    # Ceja Izquierda
                    x3, y3 = lista[295][1:]
                    x4, y4 = lista[385][1:]
                    cx2, cy2 = (x3 + x4) // 2, (y3 + y4) // 2
                    longitud2 = math.hypot(x4 - x3, y4 - y3)

                    # Boca Extremos
                    x5, y5 = lista[78][1:]
                    x6, y6 = lista[308][1:]
                    cx3, cy3 = (x5 + x6) // 2, (y5 + y6) // 2
                    longitud3 = math.hypot(x6 - x5, y6 - y5)

                    # Boca Apertura
                    x7, y7 = lista[13][1:]
                    x8, y8 = lista[14][1:]
                    cx4, cy4 = (x7 + x8) // 2, (y7 + 8) // 2
                    longitud4 = math.hypot(x8 - x7, y8 - y7)

                    # Eleccion aleatoria
                    if contador >= 10:
                        contador = 0
                        while anterior == emocion:
                            emocion = emocionAleatoria()
                        anterior = emocion
                        puntuacion += 1
                        loop.update(1)

                    # Clasificación emocional
                    # Enojado
                    if emocion == 'Enojo':
                        cv2.putText(frame, emocion, (240, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        if longitud1 < 22 and longitud2 < 22 and 75 < longitud3 < 90 and longitud4 < 8:
                            contador += 1
                    # Feliz
                    elif emocion == 'Felicidad':
                        cv2.putText(frame, emocion, (240, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                        if 20 < longitud1 < 30 and 20 < longitud2 < 30 and longitud3 > 90 and 5 < longitud4 < 30:
                            contador += 1
                    # Asombrado
                    elif emocion == 'Asombro':
                        cv2.putText(frame, emocion, (240, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                        if longitud1 > 10 and longitud2 > 10 and longitud3 > 60 and longitud3 < 110 and longitud4 > 10:
                            contador += 1
                    # Triste
                    elif emocion == 'Tristeza':
                        cv2.putText(frame, emocion, (240, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                        if longitud1 > 21 and longitud1 < 30 and longitud2 > 21 and longitud2 < 30 and longitud3 > 65 and longitud3 < 115 and longitud4 < 10:
                            contador += 1
                    # Neutral
                    else:
                        cv2.putText(frame, emocion, (240, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (155, 255, 255), 3)
                        if emocion == 'Neutralidad':
                            contador += 1

                    imagen = imagenEmocion(emocion)
                    nFrame = cv2.hconcat([frame, imagen])

    cv2.imshow("Juego de las Emociones", nFrame)
    t = cv2.waitKey(1)

    if t == ord('q'):
        break
    if puntuacion >= limite:
        time.sleep(0.5)
        break

cap.release()
cv2.destroyAllWindows()
