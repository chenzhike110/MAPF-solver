import numpy as np
from simulator import Simulator, scale
from cbs.cbs import Environment, CBS

if __name__ == "__main__":
    # boundary
    static_obstacle = []
    for i in range(-1, 600//scale+2):
        static_obstacle.append((i,-1))
    for i in range(-1, 600//scale+2):
        static_obstacle.append((600//scale+2, i))
    for i in range(-1, 600//scale+2):
        static_obstacle.append((i, 600//scale+2))
    for i in range(-1, 600//scale+2):
        static_obstacle.append((-1, i))

    agents1 = []
    agents2 = []
    agent_num = 8
    env = Simulator((601,601,3),agent_num)
    env.show()
    start, target =  env.information()
    for i in start.keys():
        agent = {'start':(start[i][0], start[i][1]),'goal':(target[i][0], target[i][1]),'name':i}
        agents1.append(agent)
        agent = {'start':(target[i][0], target[i][1]),'goal':(target[i][2], target[i][3]),'name':i}
        agents2.append(agent)
    Env1 = Environment([600//scale, 600//scale],agents1,static_obstacle)
    Env2 = Environment([600//scale, 600//scale],agents2,static_obstacle)
    cbs1 = CBS(Env1)
    cbs2 = CBS(Env2)
    solution1 = cbs1.search()
    solution2 = cbs2.search()

    path_final = {}
    for i in start.keys():
        path_final[i] = [(j['x'],j['y']) for j in solution1[i]]
        path_final[i] += [(j['x'],j['y']) for j in solution2[i]]
    env.start(path_final, None, True)