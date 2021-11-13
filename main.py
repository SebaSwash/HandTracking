# Permite identificar el número según la cantidad de dedos mostrados en la cámara
import os
import cv2
import time

from handtracking_module import HandTracking

width_cam, height_cam = 640, 480

folder_path = 'imgs'
image_list = os.listdir(folder_path) # Contiene los nombres de los archivos al interior de la carpeta
overlay_list = []

for image_filename in image_list:
    image = cv2.imread(folder_path + '/' + image_filename)
    overlay_list.append(image)


# Definición de la cámara
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, width_cam) # Configuración del ancho de la ventana
cap.set(4, height_cam) # Configuración del alto de la ventana

previous_time = 0
tracker = HandTracking(detection_confidence=0.5)

while True:
    # Lectura de la cámara
    success, img = cap.read()
    img = tracker.track_hands(img)
    landmarks_list = tracker.find_position(img, draw = False)

    hand_fingers_status = [0, 0, 0, 0, 0] # pulgar, índice, medio, anillo, meñique
    total_fingers = 0 # Cantidad de dedos que se reconocen como abiertos
    target_points = [4, 8, 12, 16, 20] # Landmarks de interés definidos por Mediapipe para identificar los dedos

    if len(landmarks_list) != 0:

        # Reconocimiento del dedo pulgar (no se puede interpretar igual que los otros debido a su anatomía)
        if landmarks_list[target_points[0]][1] >= landmarks_list[target_points[1] - 1][1]:
            hand_fingers_status[0] = 1

        # Reconocimiento de dedos (abiertos / cerrados) excluyendo al pulgar
        for i in range(1, len(target_points)):
            if landmarks_list[target_points[i]][2] < landmarks_list[target_points[i] - 2][2]:
                hand_fingers_status[i] = 1
        
        # Obtiene la suma de la cantidad de dedos que se reconocen como abiertos
        total_fingers = sum(hand_fingers_status)

    # Se accede y se muestra la imagen correspondiente a la cantidad de dedos levantados según su índice en la lista
    width, height, channel = overlay_list[total_fingers].shape
    img[0:width, 0:height] = overlay_list[total_fingers]

    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time

    #cv2.putText(img, 'FPS: ' + str(int(fps)), (450, 470), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)

    cv2.imshow('Image', img)
    cv2.waitKey(1)