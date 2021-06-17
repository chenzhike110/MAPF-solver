import argparse
import os
from dueling_dqn.agent import dqn_agent
from simulator import Simulator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # env name
    parser.add_argument("--env_name", default="MAPF", type=str)
    # use cuda
    parser.add_argument("--cuda", default=True, type=bool)
    # replay buffer size
    parser.add_argument("--buffer_size", default=20000, type=int)
    # learning rate
    parser.add_argument("--lr", default=1e-4, type=float)
    # bath size
    parser.add_argument("--bath_size", default=50, type=int)
    # gamma
    parser.add_argument("--gamma", default=0.95, type= float)
    # start learning time
    parser.add_argument("--learning_starts", default=500, type=int)
    # train frequency
    parser.add_argument("--train_freq", default=2, type=int)
    # target_network_update_freq
    parser.add_argument("--target_network_update_freq", default=2, type=int)
    # target network update tau
    parser.add_argument("--tau", default=0.005, type=float)
    # use double dqn
    parser.add_argument("--use_double_net", default=True, type=bool)
    # exploration fraction
    parser.add_argument("--exploration_fraction", default=0.7, type=int)
    # random exploration init ratio
    parser.add_argument("--init_ratio", default=0.7, type=float)
    # random exploration final ratio
    parser.add_argument("--final_ratio", default=0.1, type=float)
    # max time steps
    parser.add_argument("--total_timesteps", default=5e5, type=int)
    # save dir
    parser.add_argument("--save_dir", default='./models', type=str)
    # save model frequency
    parser.add_argument("--display_interval", default=500, type=int)
    # load model to continue
    parser.add_argument("--load_model", default=False, type=bool)
    args = parser.parse_args()

    env = Simulator((601,601,3),5)
    model = dqn_agent(env, args)
    if args.load_model:
        model_path = os.path.join(args.save_dir, args.env_name)
        model.load_dict(model_path+"/model.pt")
    model.learn_one()
