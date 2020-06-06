import cv2 as cv
import os
import numpy as np
from math import pi,sin,cos

def images_gen(name):
    year = name[13]
    ref_dir = "/extend/14-17_2500_radar/1"+str(year)+"_2500_radar"
    ref = np.fromfile(os.path.join(ref_dir, name +".ref"),
                        dtype=np.uint8).reshape(700, 900)
    ref[ref > 75] = 0
    ref[ref<45] = 0
    ref[ref!=0] = 255
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(20,20))
    closed_img = cv.morphologyEx(ref,cv.MORPH_CLOSE,kernel)
    return closed_img

def remove_small(im,threshold):
    binary = im
    # binary[binary!=0] = 255
    contours, hierarch = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        area = cv.contourArea(contours[i])
        if area < threshold:
            cv.drawContours(binary, [contours[i]], 0, 0, -1)
    #cv.imwrite("bin.png",binary)
    return binary

def skeleton(im):
    element = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    skel = np.zeros(im.shape, np.uint8)
    erode = np.zeros(im.shape, np.uint8)
    temp = np.zeros(im.shape, np.uint8)
    i = 0
    while True:
        erode = cv.erode(im, element)
        temp = cv.dilate(erode, element)
        # 消失的像素是skeleton的一部分
        temp = cv.subtract(im, temp)
        # cv.imshow('skeleton part %d' % (i,), temp)
        skel = cv.bitwise_or(skel, temp)
        im = erode.copy()
        if cv.countNonZero(im) == 0:
            break
        i+=1
    skel[skel!=0] = 255
    cv.imwrite("skel.png",skel)
    exit()

def to_thin(image, array):
    (height, width) = image.shape
    i_thin = image
    for h in range(height):
        for w in range(width):
            if image[h, w] == 0:
                a = [1] * 9
                for i in range(3):
                    for j in range(3):
                        if -1 < h-1+i < height and -1 < w-1+j < width and i_thin[h-1+i, w-1+j] == 0:
                            a[j*3+i] = 0
                sum = a[0]*1 + a[1]*2 + a[2]*4 + a[3]*8 + a[5]*16 + a[6]*32 + a[7]*64 + a[8]*128
                i_thin[h, w] = array[sum] * 255
    cv.imwrite("thin.png",i_thin)
    return i_thin


if __name__ == '__main__':
    # [:85] 1st
    # [86:142] 2nd
    # [142:222] 3rd
    files = sorted(os.listdir("./my_unet/data/result"))[:85]
    # files = [files]
    center_files = []
    points = np.ones((700,900))*255
    cnt = 0
    for file in files:
        center_file = []
        name = file.split('.')[0]
        if name[-1]!='0': continue
        print(name)
        file = cv.imread(os.path.join("./my_unet/data/result",file),0)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        copy = cv.morphologyEx(file, cv.MORPH_CLOSE, kernel)
        # img = cv.GaussianBlur(copy, (3, 3), 0)
        img = copy
        img[img!=0] = 255
        cv.imwrite("0.png",img)
        img = remove_small(img,1000)
        cv.imwrite("1.png",img)
        # skeleton(img)
        # img = to_thin(img,array)
        x = img.copy()
        x[x>0] = 1
        # img = zhangSuen(x)
        # cv.imwrite("./test.png",img)
        # exit()
        contours, hierarch = cv.findContours(img.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        threshold = 2000
        for contour in contours:
            area = cv.contourArea(contour)
            if area < threshold: continue
            # ellipse = cv.fitEllipse(contour)
            # ellipse = np.array(ellipse)
            # ellipse = [ellipse[0][0],ellipse[0][1],ellipse[1][0],ellipse[1][1],ellipse[2]]
            # print("ellipse",ellipse)
            # if max(ellipse[2],ellipse[3])<100: continue
            # print(ellipse)
            # exit()
            M = cv.moments(contour)
            center_x = int(M["m10"]/M["m00"])
            center_y = int(M["m01"]/M["m00"])
            print(center_x,center_y)
            # center_x,center_y = int(ellipse[0]),int(ellipse[1])
            if center_x != 0 and center_y != 0:
                center_file.append((center_x,center_y))
            cv.circle(points,(center_x,center_y),1,[0,0,0,255],-1)
        print("center_files:",center_files)
        # print("center_file:",center_file)
            # break
        i = 0
        while i < len(center_files):
            x,y = int(center_files[i][-1][0]),int(center_files[i][-1][1])
            l = len(center_files[i])
            d = []
            for j in range(len(center_file)):
                # print(center_file[j])
                xf,yf = int(center_file[j][0]),int(center_file[j][1])
                d.append((xf-x)**2+(yf-y)**2)
            if len(d)!=0:
                min_ = d.index(min(d))
                if d[min_]<1000:
                    center_files[i].append(center_file[min_].copy())
                    center_file[min_][:2] = [0, 0]
            if len(center_files[i])==l:
                if l>5:
                    np.savetxt(os.path.join("./datas/%d.txt")%cnt,
                           np.around(np.asarray(center_files[i]), decimals=1), fmt='%s',newline='\n')
                    cnt+=1
                center_files = center_files[:i]+center_files[i+1:]
                i-=1
            i+=1
        for j in range(len(center_file)):
            if center_file[j][0]!=0 and center_file[j][1]!=0:
                center_files.append([center_file[j]])

    for cf in center_files:
        if len(cf)<5:continue
        centers = np.asarray(cf)
        # print(centers)
        np.savetxt(os.path.join("./datas/%d.txt") % cnt, np.around(cf, decimals=1), fmt='%s', newline='\n')
        cnt+=1
    print(cnt)
    cv.imwrite("./0-85.png",points)
