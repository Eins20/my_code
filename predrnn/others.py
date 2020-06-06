import numpy as np
import cv2 as cv

if __name__ == '__main__':
    img = np.ones((700,900),np.uint8)*255
    center,size,angle = (524,233),(14,47),120
    center1,size1,angle1 = (534,224),(17,44),120

    cv.ellipse(img,center,size,angle,0,360,0,5)
    cv.ellipse(img,center1,size1,angle1,0,360,0,3)
    cv.imwrite("e1.png",img)