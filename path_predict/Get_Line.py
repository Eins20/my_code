import cv2 as cv
import Pre_Now_Lists
import args
import os
from copy import deepcopy
import numpy as np
from queue import Queue

class find_skeleton:

    def __init__(self):
        pass
    def neighbours(self,x, y, image):
        img = image
        x_1, y_1, x1, y1 = x - 1, y - 1, x + 1, y + 1
        return [img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1],  # P2,P3,P4,P5
                img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1]]  # P6,P7,P8,P9

    # 计算邻域像素从0变化到1的次数
    def transitions(self,neighbours):
        n = neighbours + neighbours[0:1]  # P2,P3,...,P8,P9,P2
        return sum((n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]))  # (P2,P3),(P3,P4),...,(P8,P9),(P9,P2)

    # Zhang-Suen 细化算法
    def zhangSuen(self,image):
        Image_Thinned = image.copy()  # Making copy to protect original image
        Image_Thinned[Image_Thinned!=0] = 1
        rows, columns = Image_Thinned.shape
        changing1 = changing2 = 1
        while changing1 or changing2:  # Iterates until no further changes occur in the image
            # Step 1
            changing1 = []
            for x in range(1, rows - 1):
                for y in range(1, columns - 1):
                    P2, P3, P4, P5, P6, P7, P8, P9 = n = self.neighbours(x, y, Image_Thinned)
                    if (Image_Thinned[x][y] == 1 and  # Condition 0: Point P1 in the object regions
                            2 <= sum(n) <= 6 and  # Condition 1: 2<= N(P1) <= 6
                            self.transitions(n) == 1 and  # Condition 2: S(P1)=1
                            P2 * P4 * P6 == 0 and  # Condition 3
                            P4 * P6 * P8 == 0):  # Condition 4
                        changing1.append((x, y))
            for x, y in changing1:
                Image_Thinned[x][y] = 0
            # Step 2
            changing2 = []
            for x in range(1, rows - 1):
                for y in range(1, columns - 1):
                    P2, P3, P4, P5, P6, P7, P8, P9 = n = self.neighbours(x, y, Image_Thinned)
                    if (Image_Thinned[x][y] == 1 and  # Condition 0
                            2 <= sum(n) <= 6 and  # Condition 1
                            self.transitions(n) == 1 and  # Condition 2
                            P2 * P4 * P8 == 0 and  # Condition 3
                            P2 * P6 * P8 == 0):  # Condition 4
                        changing2.append((x, y))
            for x, y in changing2:
                Image_Thinned[x][y] = 0
        return Image_Thinned

#longest
class prune_skeleton_beiyong:

    def __init__(self,skeleton):
        self.path = []
        self.skeleton = skeleton
        self.dirs=[(0,1),(1,0),(0,-1),(-1,0),(-1,-1),(1,-1),(-1,1),(1,1)]

    def neighbours(self,x, y, image):
        img = image
        x_1, y_1, x1, y1 = x - 1, y - 1, x + 1, y + 1
        return [img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1],  # P2,P3,P4,P5
                img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1]]  # P6,P7,P8,P9

    def mark(self,maze, pos):  # 给迷宫maze的位置pos标"2"表示“到过了”
        maze[pos[0]][pos[1]] = 2

    def passable(self,maze, pos):  # 检查迷宫maze的位置pos是否可通行
        return maze[pos[0]][pos[1]] == 1

    def find_path(self,maze, pos, end):
        self.mark(maze, pos)
        if pos == end:
            # print(pos, end=" ")  # 已到达出口，输出这个位置。成功结束
            self.path.append(pos)
            return True
        for i in range(8):  # 否则按四个方向顺序检查
            nextp = pos[0] + self.dirs[i][0], pos[1] + self.dirs[i][1]
            # 考虑下一个可能方向
            if self.passable(maze, nextp):  # 不可行的相邻位置不管
                if self.find_path(maze, nextp, end):  # 如果从nextp可达出口，输出这个位置，成功结束
                    self.path.append(pos)
                    return True

    def prune_line(self):  # 0/1
        one_points = []
        w, h = args.shape
        for i in range(1,w-1):
            for j in range(1,h-1):
                if self.skeleton[i][j]==1:
                    n = self.neighbours(i,j,self.skeleton)
                    if sum(n)==1:
                        one_points.append((i,j))

        longest = 0
        # print("total:",len(one_points))
        for i in range(len(one_points)):
            for j in range(len(one_points)):
                dis = (one_points[i][0]-one_points[j][0])**2+(one_points[i][1]-one_points[j][1])**2
                if dis>longest:
                    longest = dis
                    be = (one_points[i],one_points[j])
        self.find_path(self.skeleton,be[0],be[1])

        x = np.zeros(args.shape)

        for point in self.path:
            x[point[0]][point[1]] = 255
        return x

class prune_skeleton:

    def __init__(self,skeleton):
        self.skeleton = skeleton

    def neighbours(self,x, y, image):
        img = image
        x_1, y_1, x1, y1 = x - 1, y - 1, x + 1, y + 1
        return [img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1],  # P2,P3,P4,P5
                img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1]]  # P6,P7,P8,P9

    def getpos(self,x,y,i):
        x_1, y_1, x1, y1 = x - 1, y - 1, x + 1, y + 1
        return [(x_1, y), (x_1, y1), (x, y1), (x1, y1),  # P2,P3,P4,P5
               (x1, y), (x1, y_1), (x, y_1), (x_1, y_1)][i]  # P6,P7,P8,P9

    def get_route(self,froms, final):
        res = [final]
        x, y = final
        while froms[x][y] != (-1,-1):
            # print((x, y), end=" ")
            x, y = froms[x][y]
            res.append((x,y))
        # print((x,y))
        return res[::-1]

    def bfs(self,point):
        w,h = args.shape
        visited = [[0 for j in range(h)] for i in range(w)]
        visited[point[0]][point[1]] = 1
        froms = [[(-2, -2) for j in range(h)] for i in range(w)]
        froms[point[0]][point[1]] = (-1,-1)
        q = Queue()
        q.put(point)
        # routes = [point] # 保存队列
        route_len = [0]
        max_x, max_y = point
        max_len = 0
        w = 0 # 当前第几个位置
        while not q.empty():
            p = q.get()
            tmp_len = route_len[w]
            for i in range(8):
                pos = self.getpos(p[0], p[1], i)
                if visited[pos[0]][pos[1]] == 0 and self.skeleton[pos[0]][pos[1]] == 1:
                    q.put(pos)
                    froms[pos[0]][pos[1]] = (p[0], p[1])
                    # routes.append(pos)
                    visited[pos[0]][pos[1]] = 1
                    route_len.append(tmp_len+1)
                    if tmp_len +1 > max_len:
                        max_len = tmp_len + 1
                        max_x, max_y = pos
            w += 1
        max_route = self.get_route(froms, (max_x, max_y))
        # print(max_len)
        # print(max_route)
        return max_route

    def prune_line(self):  # 0/1
        one_points = []
        w, h = args.shape
        for i in range(1,w-1):
            for j in range(1,h-1):
                if self.skeleton[i][j]==1:
                    n = self.neighbours(i,j,self.skeleton)
                    if sum(n)==1:
                        one_points.append((i,j))
        x = []
        for point in one_points:
            # print(point)
            x.append(self.bfs(point))
        max_route = max(x,key = lambda i:len(i))
        print(max_route,len(max_route))
        new = np.zeros(args.shape)
        for p in max_route:
            new[p[0]][p[1]] = 255
        return new


def getLine(area):

    img = np.zeros(args.shape)
    cv.drawContours(img,[area],0,255,-1)

    thin = find_skeleton()
    thinned = thin.zhangSuen(img)
    cv.imwrite("skeleton_test.png",thinned)
    prune = prune_skeleton(thinned)
    pruned = prune.prune_line()

    cv.imwrite("zhang_new.png",pruned)

    return pruned

def getelli(area):
    im = np.ones(args.shape)*255
    ellipse = cv.fitEllipse(area)
    cv.ellipse(im, ellipse, (255, 255, 0), 2)

if __name__ == '__main__':
    x = cv.imread("zhang.png",0)
    x[x!=0] = 1
    getLine(x)