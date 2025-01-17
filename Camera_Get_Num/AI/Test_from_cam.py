import cv2
from ultralytics import YOLO
import time

cap = cv2.VideoCapture('/dev/video2')       #Для Windows ставим 0 или 1
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)       #устанавливаем размер камеры
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)       #устанавливаем размер камеры


model = YOLO('best.pt')  # Предобученная модель

frame_to_test_fps = 0
fps = 0

while True:
    if frame_to_test_fps == 0: time_start_frames = time.time()  #Запоминаем время начала обработки каждых 10 кадров

    is_sucess, img = cap.read()
    
    if not is_sucess:       #Если не получилось получить картинку с камеры, то пропускаем текущую итерацию цикла
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     #Преобразуем картинку в оттенки серого, но оставляем систему BRG (иначе распознование не работает)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)     

    results = model(img)        #Обрабатываем картинку моделью

    for result in results:      #Рисуем прямоугольники вокруг всех найденных обьектов, подписываем их название и указываем уверенность распознования
        boxes = result.boxes.xyxy
        confs = result.boxes.conf
        classes = result.boxes.cls
        class_names = result.names
        for num, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img, str(f'{class_names[int(classes[num])]} {round(float(confs[num]), 4)}'), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)

    if frame_to_test_fps >= 10:     #По прошествию каждых 10 кадров считаем fps
        fps = frame_to_test_fps / (time.time() - time_start_frames)
        frame_to_test_fps = 0
    else:
        frame_to_test_fps += 1
    cv2.putText(img, str(f'FPS: {round(fps, 2)}'), (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)   #Выводим fps

    cv2.imshow('out', img)      #Выводим картинку на экран

    cv2.waitKey(1)
