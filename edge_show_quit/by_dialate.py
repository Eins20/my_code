import numpy as np
import cv2 as cv
import os
from PIL import Image
kernel_erosion_3 = np.ones([3, 3], dtype=np.uint8)
kernel_erosion_5 = np.array(
    [[0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 0, 1, 0, 0]], dtype=np.uint8)
kernel_dilation_5 = np.array(
    [[0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0]], dtype=np.uint8)
kernel_dilation_7 = np.array(
    [[0, 0, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1],
     [0, 0, 1, 1, 1, 0, 0]], dtype=np.uint8)

color = {
    0: [0, 0, 0, 0,0],
    1: [0, 236, 236,180],
    2: [0, 200, 0,180],
    3: [1, 0, 246,180],
    4: [255, 255, 0,180],
    5: [231, 192, 0,180],
    6: [255, 0, 0,180],
    7: [0, 0, 0,0]
}
def judgeindex2(list):
    #print("list",list)
    if list[0]<2:
        if list[1]==0:
            if list[2]==0:
                return 0
            else:return 3
        elif list[1]==236: return 1
        elif list[1]==160: return 2
        elif list[1]==239: return 4
        elif list[1]==200: return 5
        elif list[1]==144: return 6
        else:
            print("<2",list)
            return 0
    elif list[0]==255:
        if list[1]==255:
            if list[2]==0:return 7
            elif list[2]==255:return 15
            else:
                print("255")
                return 0
        elif list[1]==144:return 9
        elif list[1]==0:
            if list[2]==0:return 10
            elif list[2]==255:return 13
            else:
                print("3")
                return 0
        else:
            print("4")
            return 0
    elif list[0]==231: return 8
    elif list[0]==166: return 11
    elif list[0]==101: return 12
    elif list[0]==153: return 14
    else: return 0

edge_color = [0,0,0,255]
def judgeindex(number):
    index = 0
    if number == 0: index = 0
    if 0 < number < 10: index = 1
    if 10 <= number < 25: index = 2
    if 25 <= number < 50: index = 3
    if 50 <= number < 100: index = 4
    if 100 <= number < 250: index = 5
    if 250 <= number < 500: index = 6
    if number > 500: index = 7
    return index

def judgenumber(index):
    numbers = [0, 10, 25, 50, 100, 250, 500]
    number = numbers[index]
    return number

def find_edge(img):
    blurred = cv.GaussianBlur(img, (3, 3), 0)
    edges = cv.Canny(blurred, 50, 150)
    return edges

#remove small district whose area <threshold
def remove_small(file,threshold,index_threshold):
    #binary = readref(filename,index=2,bi=True)#ref图像二值化
    #name = filename.split("/")[-1].split(".")[-2]
    #cv.imwrite(os.path.join(outdir, "test" + ".png"), binary)
    #exit()
   # _, binary = cv.threshold(img, 1, 255, cv.THRESH_BINARY)
    h,w,_ = file.shape
    #print(file.shape)
    binary = np.zeros((h,w),dtype=np.uint8)
    #print(binary.shape)
    for i in range(h):
        for j in range(w):
            #print(file[i,j])
            # try:print(binary[i,j])
            # except:print("hi",i,j)
            # print("done")
            binary[i,j] = judgeindex2(file[i, j])
            if binary[i,j]<index_threshold:
                binary[i,j] = 0
            else:
                binary[i,j] = 1
    contours, hierarch = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        area = cv.contourArea(contours[i])
        if area < threshold:
            cv.drawContours(binary, [contours[i]], 0, 0, -1)
    #cv.imwrite("bin.png",binary)
    return binary


def images_gen(true_image_dir,pred_image_dir):
    path_true = true_image_dir
    outdir_true = "./static/images"
    path_our = pred_image_dir
    outdir_our = "./static/edged_images"
    for file in os.listdir(path_true):
        if(file.endswith(".png")):
            #print(os.path.join(path_true, file))
            true_image = cv.imread(os.path.join(path_true, file),cv.IMREAD_UNCHANGED)
            true_image = cv.cvtColor(true_image, cv.COLOR_RGB2BGRA)
            #print("true_image",true_image)
            our_image = cv.imread(os.path.join(path_our, file),cv.IMREAD_UNCHANGED)  # 我们的预测图像
            #print("our_image",our_image)
            our_image=cv.cvtColor(our_image,cv.COLOR_RGB2BGRA)

            removed_img = remove_small(true_image,threshold=500,index_threshold=7)
            img = removed_img
            cv.imwrite("img.png", img)
            erosion1 = cv.erode(img, kernel=kernel_erosion_3, iterations=1)
            dilation1 = cv.dilate(erosion1, kernel=kernel_dilation_7, iterations=6)
            dilation2 = cv.dilate(dilation1, kernel=kernel_dilation_5, iterations=3)
            erosion2 = cv.erode(dilation2, kernel=kernel_erosion_5, iterations=6)
            # print("erosion2 == 1", erosion2[erosion2 == 1].shape)
            pic1 = erosion2 * 255

            edge7 = find_edge(pic1)  # (700,900) 0或255的二值图像
            cv.imwrite("edge7.png", edge7)

            removed_img = remove_small(true_image,threshold=500,index_threshold=9)
            img = removed_img
            erosion1 = cv.erode(img, kernel=kernel_erosion_3, iterations=1)
            dilation1 = cv.dilate(erosion1, kernel=kernel_dilation_7, iterations=6)
            dilation2 = cv.dilate(dilation1, kernel=kernel_dilation_5, iterations=3)
            erosion2 = cv.erode(dilation2, kernel=kernel_erosion_5, iterations=6)
            # print("erosion2 == 1", erosion2[erosion2 == 1].shape)
            pic1 = erosion2 * 255

            edge9 = find_edge(pic1)  # (700,900) 0或255的二值图像
            cv.imwrite("edge9.png", edge9)

            h,w = edge7.shape
            for i in range(h):
                for j in range(w):
                    if edge7[i,j]==255:
                        true_image[i,j] = edge_color
                        our_image[i,j] = edge_color
                    # if edge9[i,j]==255:
                    #     true_image[i,j] = edge_color
                    #     our_image[i,j] = edge_color
            true_image = Image.fromarray(true_image)
            true_image.save(os.path.join(outdir_true, file), "PNG")
            our_image = Image.fromarray(our_image)
            our_image.save(os.path.join(outdir_our, file), "PNG")
            print(file)


if __name__ == '__main__':
    images_gen("./ref","./our_pred")
'''
12.27(Fri.) 11:12am
done this file.
'''
'''
12.30(Mon.) 13:03
find some test pictures into ./our_pred and resize to 700,900,maybe
'''