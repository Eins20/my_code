import cv2 as cv
import numpy as np
import os
for file in os.listdir("../datas"):
    if file.endswith('png'):continue
    # if int(file.split('.')[0])<=13:continue
    x = np.zeros((700, 900)) * 255
    points = np.loadtxt(os.path.join("../datas",file))
    old = [-1,-1]
    for a,b,c,d,e in points[1:]:
        a = int(a)
        b = int(b)
        cv.ellipse(x, (int(a),int(b)), (int(c),int(d)), int(e), 0, 360, (255, 255, 255), 3)
        if old[0]!=-1:
            cv.line(x,old,(a,b),[255,255,255],1)
        old = (a,b)
    cv.imwrite(os.path.join("../datas_pic",file.split('.')[0]+".png"),x)