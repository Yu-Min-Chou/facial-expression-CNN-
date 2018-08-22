import cv2 as cv
import glob
import os
from PIL import Image

face_cascade = cv.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml')

for number in range(1,8):
    num = number
    dst = "train_data2/00" + str(num) + '_cut/'
    all_file = glob.iglob("train_data2/00"+str(num)+"/*.jpg")
    for picture in all_file:
        cvimg = cv.imread(picture)
        img = Image.open(picture)
        gray = cv.cvtColor(cvimg, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        print(picture[16:])
        for(x,y,w,h) in faces:
            area = (x,y,w+x,h+y)
            cropped_img = img.crop(area)
            cropped_img.save(dst+picture[16:],'JPEG')