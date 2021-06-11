from os import stat
import cv2
import numpy as np
from copy import deepcopy
import imageio
from matplotlib import pyplot as plt
from numpy.core.fromnumeric import size

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
        self.robot_last_pos = dict()
        self.target = dict()
        self.robot_carry = dict()
        self.size = size
        self.robot_num = robot_num
        self.frames = []
        self.steps = 0
        if static != None:
            self.robot, self.target = static
        self.colours = self.assign_colour(robot_num*3)
        self.crash = []
        self.generate_map(robot_num, size)    
        # cv2.namedWindow("Factory")
        # cv2.resizeWindow('Factory', tuple(np.array(list(size)[:2])+np.array([500,200])))
    
    def generate_map(self, robot_num, size):
        """
        generate random map to increase the complexity
        """
        assert size[0]*size[1]>robot_num *scale*3
        for i in range(1,size[0]//scale):
            cv2.line(self.canvas, (scale*i,scale), (scale*i,(size[1]//scale-1)*scale), (0,0,0))
        for i in range(1,size[1]//scale):
            cv2.line(self.canvas, (scale,i*scale), ((size[0]//scale-1)*scale,i*scale), (0,0,0))
        if len(self.robot) == 0:
            pos = np.random.randint(1,size[0]//scale, size=(3*robot_num,2))
            pos = set([tuple(i) for i in pos])
            while len(pos) < 3*robot_num:
                temp = np.random.randint(1,size[0]//scale, size=(3*robot_num - len(pos),2))
                b = set([tuple(i) for i in temp])
                for i in b:
                    if i not in pos:
                        pos.add(i)
            pos = list(pos)
            for i in range(robot_num):
                self.robot[i] = (pos[i][0],pos[i][1],i)
                self.target[i] = (pos[i+robot_num][0], pos[i+robot_num][1], pos[i+2*robot_num][0], pos[i+2*robot_num][1])
        for i in range(robot_num):
            self.draw_target(self.canvas, np.array(self.target[i][2:])*scale, self.colours[i+len(self.robot)], 5)
            self.robot_carry[i] = False       

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


    def show(self, wait=True, save=None):
        frame = deepcopy(self.canvas)
        for id_, pos in self.target.items():
            cv2.rectangle(frame, tuple(np.array(self.target[id_][:2])*scale-np.array([scale//3,scale//3])), tuple(np.array(self.target[id_][:2])*scale+np.array([scale//3,scale//3])), self.colours[id_+len(self.robot)],-1)     
        for id_, pos in self.robot.items():
            cv2.circle(frame, tuple(np.array(pos)[:-1]*scale), scale//4, self.colours[id_], -1)
        cv2.imshow("Factory",frame)
        if wait:
            cv2.waitKey(0)
        else:
            cv2.waitKey(100)
        if save != None:
            self.frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    def get_state_map(self, index, show=False):
        state = np.zeros((self.size[0]//scale+1, self.size[1]//scale+1))
        for id_, pos in self.robot.items():
            if id_ == index:
                state[pos[0]][pos[1]] = -1
            else:
                state[pos[0]][pos[1]] -= -3
        for id2_, pos2 in self.target.items():
            if id2_ == self.robot[index][2]:
                if not self.robot_carry[id_]:
                    if state[pos2[0]][pos2[1]] == -1:
                        self.robot_carry[id_] = True
                        state[pos2[0]][pos2[1]] = 2
                    else:
                        state[pos2[0]][pos2[1]] += 4
                else:
                    state[pos2[2]][pos2[3]] += 4
        state = np.rot90(state, 1)
        # state = state[1:,:-1]
        # state = state[::-1]
        if show:
            self.show()
            plt.figure()
            plt.imshow(state)
            for i in range(len(state)):
                for j in range(len(state[0])):
                    c = str(state[i][j])
                    plt.text(j, i, c, va='center', ha='center')
            plt.xlim((-0.5,len(state[0])-0.5))
            plt.ylim((-0.5,len(state)-0.5))
            plt.show()
        return np.array([state])
    
    @staticmethod
    def out_of_map(pos, size):
        if pos[0] <= 0 or pos[0] >= size[0]//scale or pos[1] <= 0 or pos[1] >= size[1]//scale:
            return True
        return False

    def step(self, action):
        path = {}
        reward = np.array([-1.0 for i in action])
        done = [False for i in action]
        states = []
        end = {}
        for id_, pos in self.robot.items():
            pos2 = self.target[pos[2]]
            end[id_] = (pos2[0], pos2[1])
            if (pos[0]-pos2[0])**2 + (pos[1]-pos2[1])**2 < 1:
                end[id_] = (pos2[2], pos2[3])
            if action[id_] == 0:
                path[id_] = [(pos[0], pos[1])]
                reward[id_] -= 0.5
            elif action[id_] == 1:
                path[id_] = [(pos[0], pos[1]+1)]
                if end[id_][1] - pos[1] > 0:
                    reward[id_] += 1.5
                else:
                     reward[id_] -= 0.5
            elif action[id_] == 2:
                path[id_] = [(pos[0]-1, pos[1])]
                if end[id_][0] - pos[0] < 0:
                    reward[id_] += 1.5
                else:
                     reward[id_] -= 0.5
            elif action[id_] == 3:
                path[id_] = [(pos[0]+1, pos[1])]
                if end[id_][0] - pos[0] > 0:
                    reward[id_] += 1.5
                else:
                     reward[id_] -= 0.5
            elif action[id_] == 4:
                path[id_] = [(pos[0], pos[1]-1)]
                if end[id_][1] - pos[1] < 0:
                    reward[id_] += 1.5
                else:
                     reward[id_] -= 0.5
            if self.out_of_map(path[id_][0], self.size):
                reward[id_] -= 20
                done[id_] = True
        self.steps += 1
        if self.steps > 200:
            done[id_] = True
        self.start(path, None, False)
        if len(self.crash) > 0:
            for i in self.crash:
                reward[i[0]] -= 20
                reward[i[1]] -= 20
                done[i[0]] = True
                done[i[1]] = True
        for id_ in self.robot.keys():
            state = self.get_state_map(id_, False)
            states.append(state)
            if np.math.hypot(self.robot[id_][0]-end[id_][0], self.robot[id_][1]-end[id_][1])<1:
                reward[id_] += 35
            if np.math.hypot(self.robot[id_][0]-self.target[id_][2], self.robot[id_][1]-self.target[id_][3]) < 1 and np.math.hypot(self.target[id_][0]-self.target[id_][2], self.target[id_][1]-self.target[id_][3]) < 1:
                reward[id_] += 35
                done[id_] = True
        return reward, np.array(states), done, {}
    
    def reset(self):
        self.crash = []
        self.canvas = np.ones(self.size, np.uint8)*255
        self.robot = {}
        self.robot_carry = {}
        self.target = {}
        self.steps = 0
        states = []
        self.generate_map(self.robot_num, self.size)
        for id_ in self.robot.keys():
            state = self.get_state_map(id_)
            self.robot_carry[id_] = False
            states.append(state)
        return np.array(states)

    def crash_check(self):
        """
        check if there are any collision
        """
        for id_, pos in self.robot.items():
            lastmiddle1 = ((self.robot_last_pos[id_][0]+pos[0])/2, (self.robot_last_pos[id_][1]+pos[1])/2)
            for id2_, pos2 in self.robot.items():
                if id_ >= id2_:
                    continue
                lastmiddle = ((self.robot_last_pos[id2_][0]+pos2[0])/2, (self.robot_last_pos[id2_][1]+pos2[1])/2)
                # print(lastmiddle, lastmiddle1, np.math.hypot(lastmiddle1[0]-lastmiddle[0],lastmiddle1[1]-lastmiddle[1]))
                if np.math.hypot(pos[0]-pos2[0], pos[1]-pos2[1]) < 1 or np.math.hypot(lastmiddle1[0]-lastmiddle[0],lastmiddle1[1]-lastmiddle[1])<=0.5:
                    self.crash.append((id_,id2_))
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

    def start(self, path, save_gif=None, wait=False):
        try:
            i = 0
            while True:
                self.robot_last_pos = self.robot.copy()
                for id_ in path:
                    if i >= len(path[id_]) or np.math.hypot(path[id_][i][0]-self.robot[id_][0], path[id_][i][1]-self.robot[id_][1]) > 1.4:
                        continue
                    cv2.line(self.canvas, tuple(np.array(self.robot[id_][:2])*scale), tuple(np.array(path[id_][i])*scale), self.colours[id_],5)
                    if self.robot[id_][2] >= 0:
                        if self.target[self.robot[id_][2]][:2]==self.robot[id_][:2]:
                            self.robot[id_] = tuple(np.append(np.array(path[id_][i]),self.robot[id_][2]))
                            self.target[self.robot[id_][2]] = tuple([self.robot[id_][0], self.robot[id_][1], self.target[self.robot[id_][2]][2], self.target[self.robot[id_][2]][3]])
                        else:
                            self.robot[id_] = tuple(np.append(np.array(path[id_][i]),self.robot[id_][2]))
                    else:
                        self.robot[id_] = tuple(np.append(np.array(path[id_][i]),self.robot[id_][2]))
                        self.carry_check()
                if self.crash_check():
                    frame = np.ones(self.size, np.uint8)*255
                    cv2.putText(frame, "Crash", (self.size[0]//2-int(2.5*scale), self.size[1]//2), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 2)
                    cv2.imshow("Factory",frame)
                    cv2.waitKey(1000)
                    break    
                self.show(wait, save_gif)
                i += 1
                if i >= max([len(i) for i in path.values()]):
                    # print("over")
                    break
        except Exception as err:
            print(err)
        if save_gif!=None:
            with imageio.get_writer("./image/"+save_gif, mode="I") as writer:
                for idx, frame in enumerate(self.frames):
                    writer.append_data(frame)
        cv2.waitKey(1)
        # cv2.destroyAllWindows()


if __name__ == "__main__":
    # # random initialize
    # env1 = Simulator((601,601,3),5)

    # # given state
    # static_origin = [{0:(1,1,1),1:(2,2,-1),2:(3,3,-1)}, {0:(8,5,7,3),1:(10,8,9,9),2:(5,10,11,2)}]
    # env2 = Simulator((601,601,3),3,static_origin)
    # env2.show()
    # state = env2.get_state_map(0, True)
    # # display

    # # get start and target
    # print(env2.information())

    # # given a path and show
    # static_origin = [{0:(1,1,0)},{0:(1,4,2,6)}]
    # path = {0:[(1,2),(1,3),(1,4),(2,4),(2,5),(2,6)]}
    # env = Simulator((601,601,3),1,static_origin)
    # env.start(path)

    # check collision
    # static_origin = [{0:(1,1,0),1:(1,3,1)},{0:(1,4,2,6),1:(10,8,9,7)}]
    # path = {0:[(1,2),(1,3),(1,4),(2,4),(2,5),(2,6)],1:[(1,3),(1,2)]}
    # env3 = Simulator((601,601,3),2,static_origin)
    # env3.start(path, None, True)

    # # check state map
    # static_origin = [{0:(1,1,0)},{0:(1,4,2,6)}]
    # action = [1,1,1,2,3,3]
    # env = Simulator((601,601,3),1,static_origin)
    # for i in action:
    #     reward, states, done, _ = env.step([i])
    #     print("reward:",reward)
    #     if done:
    #         print("done")
    #         break
    
    # check state map2
    static_origin = [{0:(1,1,0),1:(1,3,1)},{0:(1,4,2,6),1:(10,8,9,7)}]
    # path = {0:[(1,2),(1,3),(1,4),(2,4),(2,5),(2,6)],1:[(1,4),(2,4),(2,5),(2,6)]}
    # action = [[1,1],[1,3],[1,1],[3,1],[1,0],[1,0]]
    action = [[1,3],[4,3],[1,3],[4,1],[1,0],[4,0]]
    env = Simulator((601,601,3),2,static_origin)
    for i in action:
        reward, states, done, _ = env.step(i)
        print("reward:",reward)
        if np.array(done).any():
            print("done")
            break

