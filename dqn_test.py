import argparse
import os
import torch
import numpy as np
from dueling_dqn.simple_agent import dqn_agent
from dueling_dqn import utils
from simulator import Simulator
from allocate import get_allocate_matrix

def target_policy(state):
    prob = np.ones(5)/100
    # stay, down, left, right, up
    if state[0] < 0:
        prob[2] += 5*abs(state[0])
    elif state[0] > 0:
        prob[3] += 5*abs(state[0])
    if state[1] > 0:
        prob[1] += 5*abs(state[1])
    elif state[1] < 0:
        prob[4] += 5*abs(state[1])
    if state[0] == 0 and state[1] == 0:
        if sum(state[2:]) == 0:
            prob = np.zeros(5)
        prob[0] += 1
    if state[2] == 1:
        prob[2] = 0
    if state[3] == 1:
        prob[3] = 0
    if state[4] == 1:
        prob[4] = 0
    if state[5] == 1:
        prob[1] = 0
    prob = prob/sum(prob)
    action = np.random.choice(range(5), p = prob)
    return action

def main():
    parser = argparse.ArgumentParser()
    # env name
    parser.add_argument("--env_name", default="MAPF", type=str)
    # use cuda
    parser.add_argument("--cuda", default=True, type=bool)
    # load model to continue
    parser.add_argument("--load_model", default=True, type=bool)
    # save dir
    parser.add_argument("--save_dir", default='./models', type=str)
    args = parser.parse_args()

    env = Simulator((601,601,3),1)
    robots, targets = env.information()
    pairs = get_allocate_matrix(robots, targets)
    env.update_pairs(pairs)
    model = dqn_agent(env, args)
    if args.load_model:
        model_path = os.path.join(args.save_dir, args.env_name)
        model.load_dict(model_path+"/model3.538501262664795.pt")
    obs = env.reset(True)
    obs = [obs[0]]
    done = False
    while not done:
        with torch.no_grad():
            obs_tensor = model._get_tensors(obs)
            action_value = model.net(obs_tensor)
        action = utils.select_action(action_value, 0.1)
        if type(action) == np.int64 or type(action) == int:
            action = [action]
        reward, obs_, done, _ = env.step_test(action, True, "single_DQN_test.gif")
        obs = [obs_[0]]
        done = np.array(done).any()

def main2():
    parser = argparse.ArgumentParser()
    # env name
    parser.add_argument("--env_name", default="MAPF", type=str)
    # use cuda
    parser.add_argument("--cuda", default=True, type=bool)
    # load model to continue
    parser.add_argument("--load_model", default=True, type=bool)
    # save dir
    parser.add_argument("--save_dir", default='./models', type=str)
    args = parser.parse_args()

    robot_num = 25
    env = Simulator((601,601,3),robot_num)
    
    model = dqn_agent(env, args)
    if args.load_model:
        model_path = os.path.join(args.save_dir, args.env_name)
        model.load_dict(model_path+"/model3.538501262664795.pt")
    obs = env.reset(True)
    robots, targets = env.information()
    pairs = get_allocate_matrix(robots, targets)
    env.update_pairs(pairs)
    env_2 = Simulator((601,601,3), robot_num, [robots.copy(), targets.copy()],'final')
    done = False
    while not done:
        action = np.zeros(robot_num)
        for i in range(robot_num):
            action_temp = np.zeros(robot_num)
            robot_action = target_policy(obs[i])
            action_temp[i] = robot_action
            reward, obs_, done, _ = env_2.step_test(action_temp, True)
            obs = obs_
            action[i] = robot_action
        reward, obs_, done, _ = env.step_test(action, True, "Multi_DQN_test.gif")
        obs = obs_
        done = np.array(done).all()

if __name__ =="__main__":
    main2()