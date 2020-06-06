import cv2 as cv
import numpy as np
import os
from PIL import Image
import shutil

def func(img):
    print(img)
    if img[0]==0 and img[1]==0 and img[2]==0:
        return True
    else: return False

def new_pic():
    dir = "./data/result"
    for file in os.listdir(dir):
        # print(file,file[13])
        # if file[13]=='6' or file[13]=='7':
        #     write_dir = "./data/val"
        # else:
        write_dir = "./data/train"
        try:
            print(file)
            raw_img = cv.imread(os.path.join("../Unet/raw_images", file))
            cv.imwrite(os.path.join(write_dir, file), raw_img)
            img = cv.imread(os.path.join("./data/result", file))
            img = cv.resize(img, (900, 700))
            # print(img.shape)
            w, h, _ = img.shape
            temp = np.zeros((w, h), dtype=np.uint8)
            for i in range(w):
                for j in range(h):
                    if img[i, j][0] == 255 and img[i, j][1] == 255 and img[i, j][2] == 255:
                        temp[i][j] = 255
            im = Image.fromarray(temp)
            im.save(os.path.join(write_dir, file.split('.')[0] + "_mask.png"))
        except: pass
        # exit()
def mask():
    cnt = 0
    dir = "./data/896_unclose_all"
    # files = sorted(os.listdir(dir))[2000:]
    files = ['cappi_ref_201507231130_2500_0.png','cappi_ref_201506111636_2500_0.png',
             'cappi_ref_201705040736_2500_0.png','cappi_ref_201506111500_2500_0.png',
             'cappi_ref_201508100042_2500_0.png']
    for file in files:
        # if cnt>1000: return
        name = file.split('.')[0]
        if name[-1]=='0':
            write_dir = './data/val'
            srcfile = os.path.join(dir,'cappi_ref_201506111606_2500_0_mask.png')
            dstfile = os.path.join(write_dir,name+'_mask.png')
            shutil.copyfile(srcfile, dstfile)

            # srcfile = os.path.join("../Unet/raw_images",name+'.png')
            # dstfile = os.path.join(write_dir,name+'.png')
            # shutil.copyfile(srcfile, dstfile)
            # cnt+=1

if __name__ == '__main__':
    mask()



