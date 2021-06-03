from torch import dtype, int64, nn, std
import torch
import numpy as np
from torch.nn.modules import loss

def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)

def v_wrap(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)

def push_and_pull(opt, lnet, gnet, done, s_, bs, ba, br, gamma):
    if done:
        v_s_ = np.array([0. for i in range(len(s_))])
    else:
        _, v_s_ = lnet.forward(v_wrap(s_))
        v_s_ = v_s_.detach().numpy()
    
    buffer_v_target = []
    for r in br[::-1]:
        v_s_ = r +  v_s_ * gamma
        buffer_v_target.append(v_s_)
    buffer_v_target = np.array(buffer_v_target).flatten('F').reshape((-1,1))
    ba =  np.array(ba).flatten('F').reshape((-1,1))
    loss = lnet.loss_func(
        v_wrap(np.vstack(bs)),
        v_wrap(np.vstack(ba)),
        v_wrap(np.vstack(buffer_v_target))
    )

    opt.zero_grad()
    loss.backward()
    for lp, gp in zip(lnet.parameters(), gnet.parameters()):
        gp._grad = lp.grad
    opt.step()

    # pull global parameters
    lnet.load_state_dict(gnet.state_dict())
    return loss

def record(global_ep, global_ep_r, ep_r, res_queue, name, loss, step):
    with global_ep.get_lock():
        global_ep.value += 1
    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r.mean()
        else:
            global_ep_r.value = global_ep_r.value * 0.9 + ep_r.mean() * 0.1
    res_queue.put((global_ep_r.value, loss, name, step, ep_r))
    print(
        name,
        "Ep:", global_ep.value,
        "| Ep_r: %.0f" % global_ep_r.value,
    )