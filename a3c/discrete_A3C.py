import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
import torch.multiprocessing as mp

class net(nn.Module):
    def __init__(self, img_size, a_dim) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.pi1 = nn.Linear(2304, 128)
        self.pi2 = nn.Linear(128, a_dim)
        self.v1 = nn.Linear(2304, 128)
        self.v2 = nn.Linear(128, 1)
        self.distribution = torch.distributions.Categorical
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1)
        pi1 = torch.tanh(self.pi1(x))
        logits = self.pi2(pi1)
        v1 = torch.tanh(self.v1(x))
        values = self.v2(v1)
        return logits, values
    
    def choose_action(self, s):
        self.eval()
        logits, _ = self.forward(s)
        prob = F.softmax(logits, dim=1).data
        m = self.distribution(prob)
        return m.sample().numpy()[0]

    def loss_func(self, s, a, v_t):
        self.train()
        logits, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)
        
        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
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
        # self.lnet = net(N_S, N_A)           # local network
        # self.env = gym.make('CartPole-v0').unwrapped

if __name__ == "__main__":
    b = np.ones((1,10,10))
    print(b[0])
    a = [np.ones((1,10,10)) for i in range(8)]
    model = net(10,5)
    print(model(torch.Tensor(a)))
