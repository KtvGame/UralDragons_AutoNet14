import cv2
import numpy as np 
import random


SAVING_DIR = './Images5/'       #Директория сохранения, с / в конце (обязательно!)

def change_brightness_and_contrast(img, brightness, contrast):      #Изменение яркости и контрастности кадра, 0 в значении - не изменять
    img = np.int16(img)
    img = img * (contrast/127+1) - contrast + brightness
    img = np.clip(img, 0, 255)
    img = np.uint8(img)
    return img


camera = cv2.VideoCapture('/dev/video2')        #Для Windows ставим 0 или 1
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)       #устанавливаем размер камеры
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)       #устанавливаем размер камеры

image_num = 0
frame_num = 0
is_saving = False
while True:
    img = camera.read()[1]      #Читаем картинку с камеры
    
    random_brightness = random.randint(-65, 150)        #Выбираем и станваливаем случайную яркость и контрастность
    img_changed = change_brightness_and_contrast(img, random_brightness, random_brightness+random.randint(-50, 50))

    img_changed = cv2.cvtColor(img_changed, cv2.COLOR_BGR2GRAY)     #Преобразуем изображение в оттенки серого

    if frame_num % 3 == 0 and is_saving:            #Сохраняем картинку каждый 3 кадр, если сейчас сохранение не на паузе
        cv2.imwrite(f'{SAVING_DIR}{image_num}.jpg', img_changed)
        image_num += 1
        
    if is_saving: frame_num += 1

    cv2.putText(img, f'Images: {image_num}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)     #Выводим кол-во сохраненных изображений на экран
    
    #Выводим картинки на экран
    cv2.imshow('Original', img)
    cv2.imshow('Processed', img_changed)

    key = cv2.waitKey(10)
    if key == ord('s'):         #Если нажали кнопку s на клавиатуре, то начинаем/прекращаем запись кадров в файлы.
        is_saving = not is_saving