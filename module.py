import cv2 as cv
import numpy as np
def read_img(img_path):
    img=cv.imread(img_path)
    cv.imshow('IMAGE', img)
    cv.waitKey(0)
    