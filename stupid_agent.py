import numpy as np
from torch import index_add
from simulator import Simulator, scale
from allocate import get_allocate_matrix


def getrange(a, b):
    if a <= b:
        return range(a+1,b+1)
    else:
        return range(a-1,b-1,-1)

def generate_path(pos, target):
    path = []
    for i in getrange(pos[0],target[0]):
        path.append((i,pos[1]))
    for i in getrange(pos[1],target[1]):
        path.append((target[0],i))
    for i in getrange(target[1],target[3]):
        path.append((target[0],i))
    for i in getrange(target[0],target[2]):
        path.append((i,target[3]))
    return path

def check_crash(pos, last_pos):
    """
    check if there are any collision
    """
    crash = []
    for i in range(len(pos)):
        lastmiddle1 = ((pos[i][0]+last_pos[i][0])/2,(pos[i][1]+last_pos[i][1])/2)
        for j in range(i+1, len(pos)):
            lastmiddle = ((pos[j][0]+last_pos[j][0])/2,(pos[j][1]+last_pos[j][1])/2)
            if np.math.hypot(pos[i][0]-pos[j][0], pos[i][1]-pos[j][1]) < 1 or np.math.hypot(lastmiddle1[0]-lastmiddle[0],lastmiddle1[1]-lastmiddle[1])<=0.5:
                crash.append((i,j))
    return crash

def check_path(path, target_):
    maxtime = max([len(path[i]) for i in path])
    lastpos = [path[i][0] for i in path]
    targetpos = target_
    for i in range(maxtime):
        pos = []
        for id_ in path.keys():
            if i >= len(path[id_]):
                pos.append(path[id_][-1])
                path[id_].append(path[id_][-1])
            else:
                pos.append(path[id_][i])
        crash = check_crash(pos, lastpos)
        print(crash)
        while len(crash) > 0:
            for pairs in crash:
                pos = deal_with_conflict(lastpos, pos, targetpos, pairs)
                print(pairs)
                for a in range(2):
                    del path[pairs[a]][i+1:]
                    target = (targetpos[pairs[a]][0], targetpos[pairs[a]][1])
                    if target in path[pairs[a]]:
                        target = (targetpos[pairs[a]][2], targetpos[pairs[a]][3])
                        target = list(target)
                        target.extend(target)
                    else:
                        target = list(target)
                        target.extend([targetpos[pairs[a]][2], targetpos[pairs[a]][3]])
                    path[pairs[a]][i] = pos[pairs[a]]
                    path[pairs[a]].extend(generate_path(pos[pairs[a]], target))
            crash = check_crash(pos, lastpos)
        lastpos = pos
    return path

def Manhattan_distance(pos1, pos2):
    return abs(pos1[0]-pos2[0]) + abs(pos1[1]-pos2[1])

def Euclidean_distance(pos1, pos2):
    return np.math.hypot(pos2[0]-pos1[0],pos2[1]-pos1[1])

def get_next_pos(pos, action):
    pos_new = pos
    if action == 1:
        pos_new = (pos[0], pos[1]+1)
    elif action == 2:
        pos_new = (pos[0]-1, pos[1])
    elif action == 3:
        pos_new = (pos[0]+1, pos[1])
    elif action == 4:
        pos_new = (pos[0], pos[1]-1)
    return pos_new

def deal_with_conflict(last_pos, pos, targetpos, crash_idx):
    best = pos
    best_score = -1000
    idx1, idx2 = crash_idx
    for i in range(0,5):
        for j in range(4,-1,-1):
            if i==0 and j==0:
                continue
            pos[idx1] = get_next_pos(last_pos[idx1], i)
            pos[idx2] = get_next_pos(last_pos[idx2], j)
            if check_crash(last_pos, pos):
                continue
            else:
                score = 1/(Euclidean_distance(pos[idx1],targetpos[idx1])+0.01)
                score += 1/(Euclidean_distance(pos[idx2],targetpos[idx2])+0.01)
                if score > best_score:
                    best = pos
                    best_score = score
    return best

if __name__ == "__main__":
    robotnum = 8
    env = Simulator((601,601,3), robotnum)
    robots, targets = env.information()
    pairs = get_allocate_matrix(robots, targets)
    env.update_pairs(pairs)
    start, target = env.information()
    path = {}
    for idx, pos in start.items():
        path[idx] = generate_path(pos, target[pos[2]])
    path = check_path(path, target)
    env.start(path, "stupid_avoid.gif", True)