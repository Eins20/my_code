shape = (700,900)
longAxis = 0 #飑线长轴最短长度
dataDir = "../my_unet/data/result"

import cv2 as cv
import os
for i in range(5,12):
    x = cv.imread(os.path.join("./test_sk/6",str(i)+".png"))
    y = cv.imread(os.path.join("./test_sk/6",str(i)+"_yuan.png"))
    z = x+y
    cv.imwrite(os.path.join("./test_sk/6",str(i)+"_add.png"),z)


