import numpy as np
import cv2 as cv
import args
from copy import deepcopy

def count_overlap(area_pre,area_now):

    total_img = np.zeros(args.shape,dtype=np.uint8)

    # print(total_img)

    cv.drawContours(total_img,[area_pre],0,255,-1)
    cv.drawContours(total_img,[area_now],0,255,-1)
    total,_ = cv.findContours(total_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    a_total = sum([cv.contourArea(tot) for tot in total])

    # cv.imwrite("./total.png",total_img)

    return a_total

def isNext(area_pre,area_now):

    a_pre = cv.contourArea(area_pre)
    a_now = cv.contourArea(area_now)
    a_total = count_overlap(area_pre,area_now)

    a_overlap = a_pre+a_now-a_total

    delta_a = abs(a_overlap)/min(a_now,a_pre)
    # print(a_pre, a_now, a_total,a_overlap,delta_a)
    return delta_a>0.5

def areaOverlap(img_pre,img_now):

    pre_now = []
    # cv.imwrite("area_pre.png",img_pre)
    # cv.imwrite("area_now.png",img_now)
    area_pre_list,_ = cv.findContours(img_pre, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    area_now_list,_ = cv.findContours(img_now, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    l1 = deepcopy(area_pre_list)
    l2 = deepcopy(area_now_list)

    del_now = []

    for i in range(len(l1)):
        area_pre = l1[i]
        for j in range(len(l2)):
            area_now = l2[j]
            if isNext(area_pre,area_now):
                pre_now.append([area_pre,area_now])
                del_now.append(j)

    temp = []
    for i in range(len(l2)):
        if i not in del_now:
            temp.append(l2[i])
    area_now_list = temp

    for area_now in area_now_list:
        pre_now.append([None,area_now])

    return pre_now