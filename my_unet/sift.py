import cv2 as cv

x = cv.imread("1.png")
print(x)
print(x.shape)
w,h,_ = x.shape
for i in range(w):
    for j in range(h):
        if x[i][j][0]==0 and x[i][j][1]==0 and x[i][j][2]==0:
            x[i][j] = [255,255,255]

cv.imwrite("x.png",x)