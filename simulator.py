import cv2
import numpy as np
from copy import deepcopy
import imageio

scale = 35

class Simulator:

    def __init__(self, size, robot_num, static=None):
        """
        Initialize simulator multi agent path finding
        robot: {index:(x,y,carry_index)}
        target: {index}:(box_x,box_y,target_x,target_y)
        """
        self.canvas = np.ones(size, np.uint8)*255
        self.robot = dict()
        self.target = dict()
        self.size = size
        self.frames = []
        if static != None:
            self.robot, self.target = static
        self.colours = self.assign_colour(robot_num*3)
        self.generate_map(robot_num, size)    
        cv2.namedWindow("Factory")
        cv2.resizeWindow('Factory', tuple(np.array(list(size)[:2])+np.array([500,200])))
    
    def generate_map(self, robot_num, size):
        """
        generate random map to increase the complexity
        """
        assert size[0]*size[1]>robot_num *scale*3
        for i in range(1,size[0]//scale):
            cv2.line(self.canvas, (scale*i,scale), (scale*i,size[1]-scale), (0,0,0))
        for i in range(1,size[1]//scale):
            cv2.line(self.canvas, (scale,i*scale), (size[0]-scale,i*scale), (0,0,0))
        if len(self.robot) == 0:
            pos = np.random.randint(1,size[0]//scale, size=(3*robot_num,2))
            for i in range(robot_num):
                self.robot[i] = (pos[i][0],pos[i][1],-1)
                self.target[i] = (pos[i+robot_num][0], pos[i+robot_num][1], pos[i+2*robot_num][0], pos[i+2*robot_num][1])
        for i in range(robot_num):
            self.draw_target(self.canvas, np.array(self.target[i][2:])*scale, self.colours[i+len(self.robot)], 5)       

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
        point1 = np.array(point)-np.array([scale//3,scale//3])
        point2 = np.array(point)+np.array([scale//3,scale//3])
        point3 = np.array(point)+np.array([scale//3,-scale//3])
        point4 = np.array(point)-np.array([scale//3,-scale//3])
        cv2.line(frame, tuple(point1), tuple(point2), color, thick)
        cv2.line(frame, tuple(point3), tuple(point4), color, thick)


    def show(self, wait=True, save=False):
        frame = deepcopy(self.canvas)
        for id_, pos in self.target.items():
            cv2.rectangle(frame, tuple(np.array(self.target[id_][:2])*scale-np.array([scale//3,scale//3])), tuple(np.array(self.target[id_][:2])*scale+np.array([scale//3,scale//3])), self.colours[id_+len(self.robot)],-1)     
        for id_, pos in self.robot.items():
            cv2.circle(frame, tuple(np.array(pos)[:-1]*scale), scale//3, self.colours[0], -1)
        cv2.imshow("Factory",frame)
        if wait:
            cv2.waitKey(0)
        else:
            cv2.waitKey(100)
        if save:
            self.frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def step(self):
        pass

    def crash_check(self):
        """
        check if there are any collision
        """
        for id_, pos in self.robot.items():
            for id2_, pos2 in self.robot.items():
                if id_ >= id2_:
                    continue
                if (pos[0]-pos2[0])**2 + (pos[1]-pos2[1])**2 < 1:
                    return True
        return False
    
    def carry_check(self):
        """
        check if the robot carry the box
        """
        for id_, pos in self.robot.items():
            if pos[2] != -1:
                continue
            for id2_, pos2 in self.target.items():
                if (pos[0]-pos2[0])**2 + (pos[1]-pos2[1])**2 < 1:
                    self.robot[id_] = tuple(np.append(np.array(self.robot[id_])[:2], id2_))
                    break
    
    def information(self):
        return self.robot, self.target

    def start(self, path, save_gif=False):
        try:
            i = 0
            while True:
                for id_ in path:
                    if i >= len(path[id_]):
                        continue
                    self.robot[id_] = tuple(np.append(np.array(path[id_][i]),self.robot[id_][2]))
                if self.crash_check():
                    frame = np.ones(self.size, np.uint8)*255
                    cv2.putText(frame, "Crash", (self.size[0]//2-int(2.5*scale), self.size[1]//2), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 2)
                    cv2.imshow("Factory",frame)
                    break
                self.carry_check()
                for j in self.robot:
                    if self.robot[j][2] >= 0:
                        self.target[self.robot[j][2]] = tuple([self.robot[j][0], self.robot[j][1], self.target[self.robot[j][2]][2], self.target[self.robot[j][2]][3]])
                self.show(False, save_gif)
                i += 1
                if i >= max([len(i) for i in path.values()]):
                    break
        except Exception as err:
            print(err)
        if save_gif:
            with imageio.get_writer("Seperate_Astar.gif", mode="I") as writer:
                for idx, frame in enumerate(self.frames):
                    writer.append_data(frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # random initialize
    env1 = Simulator((601,601,3),5)

    # given state
    static_origin = [{0:(1,1,-1),1:(2,2,-1),2:(3,3,-1)}, {0:(8,5,7,3),1:(10,8,9,9),2:(5,10,11,2)}]
    env2 = Simulator((601,601,3),3,static_origin)

    # display
    # env2.show()

    # get start and target
    print(env2.information())

    # given a path and show
    static_origin = [{0:(1,1,-1)},{0:(1,4,2,6)}]
    path = {0:[(1,2),(1,3),(1,4),(2,4),(2,5),(2,6)]}
    env = Simulator((601,601,3),1,static_origin)
    # env.start(path)

    # check collision
    static_origin = [{0:(1,1,-1),1:(1,3,-1)},{0:(1,4,2,6),1:(10,8,9,7)}]
    path = {0:[(1,2),(1,3),(1,4),(2,4),(2,5),(2,6)],1:[(1,2),(1,3),(1,4),(2,4),(2,5),(2,6)]}
    env3 = Simulator((601,601,3),2,static_origin)
    env3.start(path)