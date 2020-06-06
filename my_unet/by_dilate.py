import numpy as np
import cv2 as cv
import os
from PIL import Image

def find_edge(img):
    blurred = cv.GaussianBlur(img, (3, 3), 0)
    edges = cv.Canny(blurred, 50, 150)
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


def images_gen(true_image_dir):
    l = ['cappi_ref_201506111500_2500_0.png',
         'cappi_ref_201506111636_2500_0.png',
         'cappi_ref_201507231130_2500_0.png',
         'cappi_ref_201508100042_2500_0.png',
         'cappi_ref_201705040736_2500_0.png']
    path_true = true_image_dir
    # for file in os.listdir(path_true):
    for file in l:
        if file.split('.')[0][-1]=='0':
            print('ooo')
            colored = cv.imread(os.path.join(path_true,file),cv.IMREAD_UNCHANGED)
            #print(os.path.join(path_true, file))
            try:
                ref = np.fromfile(os.path.join("/extend/14-17_2500_radar/15_2500_radar", file.split('.')[0]+".ref"),
                                  dtype=np.uint8).reshape(700, 900)
            except:
                ref = np.fromfile(os.path.join("/extend/14-17_2500_radar/17_2500_radar", file.split('.')[0] + ".ref"),
                                  dtype=np.uint8).reshape(700, 900)
            ref[ref > 75] = 0
            ref[ref<45] = 0
            true_image = ref

            # cv.imwrite("removed_img7.png",removed_img)
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(7,7))
            closed_img = cv.morphologyEx(true_image,cv.MORPH_CLOSE,kernel)
            removed_img = remove_small(closed_img,threshold=500)
            # cv.imwrite("closed_img7.png",closed_img)
            edge7 = find_edge(removed_img)#(700,900) 0或255的二值图像
            # cv.imwrite("edge43.png",edge7)

            h,w = edge7.shape
            for i in range(h):
                for j in range(w):
                    if edge7[i,j]==255:
                        colored[i,j] = [0,0,0,255]
            cv.imwrite(os.path.join("dilate",file),colored)
            # true_image.save(os.path.join(outdir_true, file), "PNG")
            print(file)
            # exit()


if __name__ == '__main__':
    images_gen("/home/ices/work/tzh/Unet/raw_images")
'''
12.27(Fri.) 11:12am
done this file.
'''
'''
12.30(Mon.) 13:03
find some test pictures into ./our_pred and resize to 700,900,maybe
'''