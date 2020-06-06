import numpy as np
import cv2 as cv
import os
from PIL import Image
import matplotlib.pyplot as plt

def find_edge(img):
    # blurred = cv.GaussianBlur(img, (3, 3), 0)
    print(img,img.dtype,img.shape)
    edges = cv.Canny(img, 50, 150)
    return edges

#remove small district whose area <threshold
def remove_small(file,threshold):
    binary = file
    binary[binary!=0] = 255
    contours, hierarch = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        area = cv.contourArea(contours[i])
        if area < threshold:
            cv.drawContours(binary, [contours[i]], 0, 0, -1)
    #cv.imwrite("bin.png",binary)
    return binary

def color_radar(img,flag=True):
    if flag:
        img = pixel_to_dBZ(img)
        # img = mapping(img)
    else:
        pass
    return img

def pixel_to_dBZ(img):
    img = img.astype(np.float)/255.0
    img = img * 95.0
    img[img<15] = 0
    return img.astype(np.uint32)

def images_gen(path_true):
    out = np.ones((101,101))*255
    for file in sorted(os.listdir(path_true)):
        if file.split('.')[0][-1]=='0':
            # print('ooo')
            ref = Image.open(os.path.join(path_true,file)).convert('L')
            ref = np.array(ref)
            print(ref)
            ref = color_radar(ref)
            ref[ref<50] = 0

            ref = np.asarray(ref)
            print(ref)
            plt.imsave("./plt.png",ref)
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
            closed_img = cv.morphologyEx(ref,cv.MORPH_CLOSE,kernel)
            closed_img = ref
            # removed_img = remove_small(closed_img,threshold=500)
            cv.imwrite("close50.png",closed_img)
            edge7 = find_edge(closed_img)#(700,900) 0或255的二值图像
            cv.imwrite("edge50.png",edge7)

            # h,w = edge7.shape
            # for i in range(h):
            #     for j in range(w):
            #         if edge7[i,j]==255:
            #             out[i,j] = 0
            # cv.imwrite("./out.png",out)
            print(file)
            exit()


if __name__ == '__main__':
    images_gen("./data/achieve_fold/ground_truth")
'''
12.27(Fri.) 11:12am
done this file.
'''
'''
12.30(Mon.) 13:03
find some test pictures into ./our_pred and resize to 700,900,maybe
'''