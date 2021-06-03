from os import name
from matplotlib.pyplot import step
import torch.multiprocessing as mp
from simulator import Simulator
from a3c.discrete_A3C import net, Worker
from a3c.shared_adam import SharedAdam
from torch.utils.tensorboard import SummaryWriter 
import time

if __name__ == "__main__":
    mp.set_start_method('spawn')
    gnet = net(5)
    gnet.share_memory()
    writer = SummaryWriter('./a3c/Logs')
    opt = SharedAdam(gnet.parameters(), lr=1e-3, betas=(0.92,0.999))
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()
    # workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(mp.cpu_count()//2)]
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, 1)]
    [w.start() for w in workers]
    start = time.time()
    while True:
        try:
            r, loss, name, total_step = res_queue.get()
            if r is not None:
                writer.add_scalar("loss/loss_"+name, loss, global_step=total_step, walltime=None)
                writer.add_scalar("mean_reward", r, global_step=time.time()-start, walltime=None)
        except:
            break
    [w.join() for w in workers]  
    # model = net(3)
    # env = Simulator((601,601,3),2)
    # done = False
    # state = env.reset()
    # env.show()
    # while not done:
    #     action = model.choose_action(state)
    #     print(action)
    #     reward, states, done, _ = env.step(action)
    #     env.show()
