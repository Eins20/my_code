import cv2 as cv
import os
import numpy as np
from PIL import Image

#remove small district whose area <threshold
def remove_small(file,threshold):
    contours, hierarch = cv.findContours(file, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        area = cv.contourArea(contours[i])
        if area < threshold:
            cv.drawContours(file, [contours[i]], 0, 0, -1)
    #cv.imwrite("bin.png",binary)
    return file

def draw_edge(name):
    # contours, hierarch = cv.findContours(file, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    # for i in range(len(contours)):
    #     area = cv.contourArea(contours[i])
    #     if area < threshold:
    #         cv.drawContours(file, [contours[i]], 0, 0, -1)
    edges = cv.imread(os.path.join("./data/my_result",name+".png"),0)
    now = os.path.join("/home/ices/work/tzh/predrnn/results/my_data_predrnn/1050/1","pd"+str(int(name[2:])+10)+".png")
    yuan = os.path.join("/home/ices/work/tzh/predrnn/results/my_data_predrnn/1050/1",name+".png")
    raw = cv.imread((now),cv.IMREAD_UNCHANGED)
    yuan = cv.imread((yuan),cv.IMREAD_UNCHANGED)
    w,h = edges.shape
    for i in range(w):
        for j in range(h):
            if edges[i,j]:
                raw[i,j] = np.array([255])
                yuan[i,j] = np.array([255])
    cv.imwrite(now,raw)
    print(name)


if __name__ == '__main__':
    for file in os.listdir("./data/val"):
        if file.split('.')[0][-1]=='0':
            print(file)
            img = cv.imread(os.path.join("./data/train",file))
            mask = cv.imread(os.path.join("./data/train",file.split('.')[0]+"_mask.png"))
            edges = cv.Canny(mask, 50, 150)
            w, h = edges.shape
            for i in range(w):
                for j in range(h):
                    if edges[i, j]:
                        img[i, j] = np.array([0, 0, 0])
            cv.imwrite(os.path.join("./data/temp", file.split('.')[0] + ".png"), img)
            exit()

