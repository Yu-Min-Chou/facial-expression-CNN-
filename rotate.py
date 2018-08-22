import numpy as np
import argparse
import cv2 as cv
import glob
from PIL import Image

def rotate_5(image, angle = 6, center = None, scale = 1.0):
    (h,w) = image.shape[:2]

    if center is None:
        center = (w/2, h/2)

    M = cv.getRotationMatrix2D(center, angle, scale)
    rotated = cv.warpAffine(image, M, (w, h))

    return rotated

def rotate_n5(image, angle = -6, center = None, scale = 1.0):
    (h,w) = image.shape[:2]

    if center is None:
        center = (w/2, h/2)

    M = cv.getRotationMatrix2D(center, angle, scale)
    rotated = cv.warpAffine(image, M, (w, h))

    return rotated
for x in range(1,8):
    for name in glob.glob('train_data2_alignment_rotated/00'+str(x)+'/*.jpg'):
        im = cv.imread(name)
        im_5 = rotate_5(im)
        im_n5 = rotate_n5(im)
        name = str(name).rstrip(".jpg")
        cv.imwrite(name + '_ro5.jpg',im_5)
        cv.imwrite(name + '_ron5.jpg',im_n5)
