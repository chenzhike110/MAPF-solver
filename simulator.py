import cv2
import numpy as np
import random 
from copy import deepcopy

scale = 20

class Simulator:

    def __init__(self, size, robot_num):
        self.canvas = np.ones(size, np.uint8)*255
        self.robot = dict()
        self.target = dict()
        self.size = size
        self.generate_map(robot_num, size)
        self.colours = self.assign_colour(robot_num*2)
        cv2.namedWindow("Factory")
        cv2.resizeWindow('Factory', tuple(np.array(list(size)[:2])+np.array([100,100])))
    
    def generate_map(self, robot_num, size):
        assert size[0]>robot_num *30 and size[1]>robot_num*30
        for i in range(size[0]//scale+1):
            cv2.line(self.canvas, (scale*i,0), (scale*i,size[1]-1), (0,0,0))
        for i in range(size[1]//scale+1):
            cv2.line(self.canvas, (0,i*scale), (size[0]-1,i*scale), (0,0,0))
        x = random.sample(range(0, size[0]//scale), 3*robot_num)
        y = random.sample(range(0, size[1]//scale), 3*robot_num)
        for i in range(robot_num):
            self.robot[i] = (x[i],y[i])
            self.target[i] = (x[i+robot_num], y[i+robot_num], x[i+2*robot_num], y[i+2*robot_num])         

    @staticmethod
    def assign_colour(num):
        def colour(x):
            x = hash(str(x+42))
            return ((x & 0xFF, (x >> 8) & 0xFF, (x >> 16) & 0xFF))
        colours = dict()
        for i in range(num):
            colours[i] = colour(i)
        return colours
    
    @staticmethod
    def draw_target(frame, point, color, thick):
        point1 = np.array(point)-np.array([0.5,0.5])
        point2 = np.array(point)+np.array([0.5,0.5])
        point3 = np.array(point)+np.array([0.5,-0.5])
        point4 = np.array(point)-np.array([0.5,-0.5])

    def show(self):
        frame = deepcopy(self.canvas)
        # for id_, pos in self.robot.items():
        #     cv2.circle(frame, pos, 1, self.colours[id_], 1)
        #     cv2.rectangle(frame, np.array(self.target[id_][:2])-np.array([0.5,0.5]), np.array(self.target[id_][:2])+np.array([0.5,0.5]), self.colours[id_+len(self.robot)])
            # draw_target(frame, self.target[id_][2:], self.colours[id_+len(self.robot)], 1)
        cv2.resizeWindow('Factory', tuple(np.array(list(self.size)[:2])+np.array([100,100])))
        cv2.imshow("Factory",frame)
        cv2.waitKey(0)

if __name__ == "__main__":
    env = Simulator((241,241,3),3)
    env.show()