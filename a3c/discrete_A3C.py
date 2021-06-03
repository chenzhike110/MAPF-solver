from matplotlib.pyplot import winter
from .shared_adam import SharedAdam
from simulator import Simulator
import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
import torch.multiprocessing as mp

from .utils import v_wrap, weights_init, normalized_columns_initializer, push_and_pull, record

UPDATE_ITER = 200
MAX_ITER = 30000
GAMMA = 0.9

class net(nn.Module):
    def __init__(self, a_dim) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.pi1 = nn.Linear(288, 256)
        self.v1 = nn.Linear(288, 256)
        self.critic_linear = nn.Linear(256, 1)
        self.actor_linear = nn.Linear(256, a_dim)
        self.softmax = nn.Softmax(dim=1)
        self.distribution = torch.distributions.Categorical
    
    def forward(self, x):
        x = torch.Tensor(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.flatten(x, 1)
        pi1 = torch.tanh(self.pi1(x))
        logits = self.actor_linear(pi1)
        v1 = torch.tanh(self.v1(x))
        values = self.critic_linear(v1)
        return logits, values
    
    def choose_action(self, s, random=1):
        self.eval()
        logits, _ = self.forward(s)
        if random > 0 and random < 1:   
            prob = self.softmax(logits)
            prob += torch.rand(torch.Size([1,5]), out=None)*random
            m = self.distribution(prob)
            action = m.sample().numpy()
        elif random == 0:
            prob = self.softmax(logits)
            m = self.distribution(prob)
            action = m.sample().numpy()
        else:
            action = torch.argmax(logits,dim=1).numpy()
        return action

    def loss_func(self, s, a, v_t):
        self.train()
        logits, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)
        
        probs = self.softmax(logits)
        m = self.distribution(probs)
        temp1 = m.log_prob(a)
        temp2 = td.detach().squeeze()
        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        return total_loss

class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name):
        super(Worker, self).__init__()
        self.name = 'w%02i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.lnet = net(5)           # local network
        self.opt = opt
        # self.robot_num = max(int(self.name[1:])%8, 1)
        self.robot_num = 1
        self.env = Simulator((601,601,3),self.robot_num)
    
    def run(self):
        total_step = 1
        random = 1.0
        while self.g_ep.value < MAX_ITER:
            state = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = np.array([0. for i in range(self.robot_num)])
            while True:
                buffer_s.append(state)
                action = self.lnet.choose_action(state, random)
                r, state, done, _ = self.env.step(action)
                ep_r += r
                
                buffer_a.append(action)
                buffer_r.append(r)

                if total_step % UPDATE_ITER == 0 or done:
                    loss = push_and_pull(self.opt, self.lnet, self.gnet, done, state, buffer_s, buffer_a, buffer_r, GAMMA)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:
                        if self.g_ep.value <= MAX_ITER/4:
                            random = random*0.999
                        if self.g_ep.value > MAX_ITER/4:
                            random = 0
                        elif self.g_ep.value > MAX_ITER*2/3:
                            random = -1
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name, int(loss), total_step)
                        break
                
                total_step += 1
        self.res_queue.put(None)


if __name__ == "__main__":
    # b = np.ones((1,10,10))
    # print(b[0])
    # a = torch.Tensor([np.ones((1,10,10)) for i in range(8)])
    # print(model.choose_action(a))
    gnet = net(5)
    gnet.share_memory()
    opt = SharedAdam(gnet.parameters, lr=1e-4, betas=(0.92,0.999))
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(mp.cpu_count())]
    [w.start() for w in workers]
    res = []
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]           

