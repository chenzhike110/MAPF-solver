from matplotlib.pyplot import sca
from simulator import Simulator, scale
from a_star import AStarPlanner

if __name__ == "__main__":
    # two agent plan seperately
    agent_num = 5
    env = Simulator((601,601,3),agent_num)
    start, target =  env.information()
    env.show()
    # set boundary
    ox, oy = [], []
    for i in range(-1, 600//scale+2):
        ox.append(i)
        oy.append(-1.0)
    for i in range(-1, 600//scale+2):
        ox.append(600//scale+2)
        oy.append(i)
    for i in range(-1, 600//scale+2):
        ox.append(i)
        oy.append(600//scale+2)
    for i in range(-1, 600//scale+2):
        ox.append(-1.0)
        oy.append(i)
    
    planner = AStarPlanner(ox,oy,1,1)
    path = planner.get_path(0, start[0][:2], target[0][:2], target[0][2:])
    for i in range(1,agent_num):
        path.update(planner.get_path(i, start[i][:2], target[i][:2], target[i][2:]))
    env.start(path, "Seperate_A*.gif")