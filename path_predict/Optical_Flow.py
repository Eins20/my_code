import numpy as np
import os
import cv2 as cv

def optical_flow():
    pass

if __name__ == '__main__':
    x = cv.imread("pruned.png",0)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    for i in range(10):
        x = cv.dilate(x,kernel)
    cv.imwrite("dilated.png",x)
    y = cv.resize(x,(90,70))
    x = cv.resize(y,(900,700))
    cv.imwrite("nice.png",x)