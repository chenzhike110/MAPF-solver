import argparse
import os
import torch
import numpy as np
from dueling_dqn.agent import dqn_agent
from simulator import Simulator

def target_policy(state):
    prob = np.ones(5)/20
    if state[0] < 0:
        prob[3] += abs(state[0])
    elif state[0] > 0:
        prob[4] += abs(state[0])
    elif state[1] < 0:
        prob[5]

if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    # env name
    parser.add_argument("--env_name", default="MAPF", type=str)
    # use cuda
    parser.add_argument("--cuda", default=True, type=bool)
    # load model to continue
    parser.add_argument("--load_model", default=False, type=bool)
    args = parser.parse_args()

    env = Simulator((601,601,3),1)
    model = dqn_agent(env, args)
    if args.load_model:
        model_path = os.path.join(args.save_dir, args.env_name)
        model.load_dict(model_path+"model71.96770477294922.pt")
    obs = env.reset(True)
    done = False
    while not done:
        with torch.no_grad():
            obs_tensor = model._get_tensors(obs)
            action_value = model(obs_tensor)
        action = select_action(action_value, 0.1)
        if self.env.robot_num == 1:
            action = [action]
        reward, obs_, done, _ = self.env.step_test(action)
        obs = obs_
        done = np.array(done).any()