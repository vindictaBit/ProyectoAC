import cv2
import mediapipe as mp
import math

# -Realizamos la Video Captura: 0 cámara integrada / 1 cámara no integrada
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Definimos el ancho de la ventana
cap.set(4, 720)  # Definimos el alto de la ventana

# -Creamos nuestra funcion de dibujo
mpDibujo = mp.solutions.drawing_utils
ConfDibu = mpDibujo.DrawingSpec(thickness=1, circle_radius=1)  # Ajustamos la configuracion de dibujo

# Creanos un objeto donde almacenarenos la malla facial
mpMallaFacial = mp.solutions.face_mesh  # Prinero llamamos la funcion
MallaFacial = mpMallaFacial.FaceMesh(max_num_faces=1)  # Creamos el objeto(Ctrl+Click)

# --- While principal
while True:
    ret, frame = cap.read()
    # ----- Correccion de color-mad
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (0, 0), fx=1.5, fy=1.5)

    # Observanos los resultados
    resultados = MallaFacial.process(frameRGB)

    # Creamos unas listas donde almacenarenos los resultados
    px = []
    py = []
    lista = []
    r = 5
    t = 3

    if resultados.multi_face_landmarks:  # Si detectamos algun rastre
        for rostros in resultados.multi_face_landmarks:  # Mostramos el rostro detectado
            mpDibujo.draw_landmarks(frame, rostros, mpMallaFacial.FACE_CONNECTIONS, ConfDibu, ConfDibu)

            # Ahora vamos a extraer los puntos del rostro detectado
            for id, puntos in enumerate(rostros.landmark):
                # print(puntos)#Nos entrega una proporcion
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
                    # cv2.Line(frame(x1 y1),(x2 V2)(000)t)
                    # cv2.circle(frane,(x1,y1),r.(0,0,0),cv2.FILLED)
                    # cv2.circle(frame, (x2, y2), r(0 0 0), cv2.FILLED)
                    # cv2.circle(frane, (cx, cy), r.(0, 0, 0) cv2.FILLED)

                    longitud1 = math.hypot(x2 - x1, y2 - y1)
                    # print(longitud1)

                    # Ceja Izquierda
                    x3, y3 = lista[295][1:]
                    x4, y4 = lista[385][1:]
                    cx2, cy2 = (x3 + x4) // 2, (y3 + y4) // 2
                    longitud2 = math.hypot(x4 - x3, y4 - y3)
                    # print(longitud2)

                    # Boca Extremos
                    x5, y5 = lista[78][1:]
                    x6, y6 = lista[308][1:]
                    cx3, cy3 = (x5 + x6) // 2, (y5 + y6) // 2
                    longitud3 = math.hypot(x6 - x5, y6 - y5)
                    # print(longitud3)

                    # Boca Apertura
                    x7, y7 = lista[13][1:]
                    x8, y8 = lista[14][1:]
                    cx4, cy4 = (x7 + x8) // 2, (y7 + 8) // 2
                    longitud4 = math.hypot(x8 - x7, y8 - y7)
                    # print(longitud4)

                    # Clasificacion Emocional

                    # Enojado
                    if longitud1 < 19 and longitud2 < 19 and 80 < longitud3 < 95 and longitud4 < 5:
                        cv2.putText(frame, 'Persona Enojada', (480, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255), 3)
                    # Feliz
                    elif longitud1 > 20 and longitud1 < 30 and longitud2 > 20 and longitud2 < 30 and longitud3 > 109 and longitud4 > 10 and longitud4 < 20:
                        cv2.putText(frame, 'Persona Feliz', (480, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 255, 255), 3)

                    # Asombrado
                    elif longitud1 > 35 and longitud2 > 35 and longitud3 > 80 and longitud3 < 90 and longitud4 > 20:
                        cv2.putText(frame, 'Persona Asombrada', (480, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 255, 0), 3)

                    # Triste
                    elif longitud1 > 20 and longitud1 < 35 and longitud2 > 20 and longitud2 < 35 and longitud3 > 80 and longitud3 < 95 and longitud4 < 5:
                        cv2.putText(frame, 'Persona Triste', (480, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (255, 0, 0), 3)

    cv2.imshow("frame", frame)
    t = cv2.waitKey(1)

    # if t == 27:
    if t == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
