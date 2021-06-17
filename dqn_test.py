import argparse
import os
from dueling_dqn.agent import dqn_agent
from simulator import Simulator

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
    model.go()