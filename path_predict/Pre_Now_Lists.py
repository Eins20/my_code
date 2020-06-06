import os
import cv2 as cv
import numpy as np
from copy import deepcopy
from Area_Overlap import areaOverlap
from Get_Line import getLine,getelli
import args

def remove_small(img,threshold=500):
    binary = img
    contours, hierarch = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        area = cv.contourArea(contours[i])
        if area < threshold:
            cv.drawContours(binary, [contours[i]], 0, 0, -1)
    return binary

def isGoal(contour):

    ellipse = cv.fitEllipse(contour)
    long = max(ellipse[1][0],ellipse[1][1])
    # print(ellipse)
    # print(long)
    return long>args.longAxis

def preNowLists(root,begin,end):

    file_list = [os.path.join(root,filename)
                 for filename in sorted(os.listdir(root))[begin:end]]

    if len(file_list)==0:
        print("No image file!")
        exit()

    groundtruth_lists = []

    file_begin = file_list[0]
    img_begin = cv.imread(file_begin,0)
    img_begin = remove_small(img_begin)
    contours_begin,_ = cv.findContours(img_begin, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for contour in contours_begin:
        if isGoal(contour):
            # line = getLine(contour)
            groundtruth_lists.append([file_begin,contour])

    img_pre = img_begin

    for file in file_list[1:]:
        img_now = cv.imread(file,0)
        img_now = remove_small(img_now)
        pre_now = areaOverlap(img_pre,img_now)
        img_pre = img_now
        to = len(groundtruth_lists)
        #先将有多个pre的now找出并一一将pre结尾，建立新now
        flag = [True]*len(pre_now)
        for i in range(len(pre_now)):
            if not flag[i]:continue
            temp = [i]
            for j in range(i+1,len(pre_now)):
                if not flag[j]:continue
                if np.all(pre_now[j][1]==pre_now[i][1]):
                    temp.append(j)
            if len(temp)<=1:continue
            for h in temp:
                flag[h] = False
            groundtruth_lists.append([file,pre_now[i][1]])
            now_gt_index = len(groundtruth_lists)-1
            for j in temp:
                pre_gt_index = None
                ends = [g[-1] for g in groundtruth_lists]
                for en in range(len(ends)):
                    if np.all(ends[en]==pre_now[j][0]):
                        pre_gt_index = en
                        break
                groundtruth_lists[pre_gt_index].append(now_gt_index)
                groundtruth_lists[pre_gt_index].append(np.array([-1]))

        pn = deepcopy(pre_now)
        del_pn = []
        for k in range(len(pn)):# area_now无前者，乃新生,即pre = None
            area_pre,area_now = pn[k]
            try:
                if not area_pre:
                    # line = getLine(area_pre)
                    groundtruth_lists.append([file,area_now])
                    del_pn.append(k)
            except:
                pass
        tmp = []
        for k in range(len(pre_now)):
            if k not in del_pn:
                tmp.append(pre_now[k])
        pre_now = tmp


        for t in range(to):
            groundtruth = groundtruth_lists[t]
            pre = groundtruth[-1]
            if np.all(pre==np.array([-1])):
                continue
            now = []
            cp = deepcopy(pre_now)
            for i in range(len(cp)):
                area_pre, area_now = cp[i]
                if np.all(area_pre==pre):
                    now.append(area_now)
            if len(now)==0:#无后继,-1表示文件结束
                groundtruth.append(np.array([-1]))
            elif len(now)==1:#一个后继，无分叉
                # line = getLine(now[0])
                groundtruth.append(now[0])
            elif len(now)>1:#多后继，分叉, ，开后继文件，先写后继文件index，再写-1
                gt_index = len(groundtruth_lists)
                now_gts_index = []
                for no in now:
                    # line = getLine(no)
                    groundtruth_lists.append([file,no])#分叉所得，不写父亲
                    now_gts_index.append(gt_index)
                    gt_index+=1
                groundtruth.append(now_gts_index)
                groundtruth.append(np.array([-1]))

    # gt0 = groundtruth_lists[0][1:]
    # for i in range(len(gt0)):
    #     contour = gt0[i]
    #     co = np.zeros(args.shape)
    #     cv.drawContours(co, [contour], 0, 255, -1)
    #     cv.imwrite(str(i)+".png",co)
    #     print(i)
#
# #以上存的都是轮廓，写文件的时候可换成椭圆/线
    print("wirting...")
    i = 0
    for groundtruth in groundtruth_lists:
        if len(groundtruth)<8:continue
        temp = []
        if not os.path.exists("./test_sk/%d"%i):
            os.makedirs("./test_sk/%d"%i)
        # temp = [groundtruth[0]]
        for contour in groundtruth[1:]:
            if isinstance(contour, list) \
                or isinstance(contour,int) \
                or np.all(contour==np.array([-1])):
                break
            temp.append(getelli(contour))
        for img_index in range(len(temp)):
            cv.imwrite(os.path.join("./test_sk",str(i),str(img_index)+'.png'),temp[img_index])
            # yuan = np.ones(args.shape)*255
            # cv.drawContours(yuan, [groundtruth[img_index+1]], 0, 0, -1)
            # cv.imwrite(os.path.join("./test_sk",str(i),str(img_index)+'_yuan.png'),yuan)
        # np.savetxt(os.path.join("./datas/%d.txt") % i, temp, fmt='%s', newline='\n')
        i+=1
    print("done! total %d"%i)