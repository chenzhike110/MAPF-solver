from dueling_dqn.agent import dqn_agent
import argparse
from simulator import Simulator
from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # env name
    parser.add_argument("--env_name", default="MAPF", type=str)
    # use cuda
    parser.add_argument("--cuda", default=True, type=bool)
    # load model to continue
    parser.add_argument("--load_model", default=False, type=bool)
    args = parser.parse_args()

    env = Simulator((601,601,3),10)
    states = env.reset()
    model = dqn_agent(env, args)
    writer = SummaryWriter('runs/dueling_double_dqn')
    writer.add_graph(model.net, model._get_tensors(states))
    # writer.add_graph(model.target_net, model._get_tensors(states))
    writer.close()