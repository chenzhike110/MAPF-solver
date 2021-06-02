from torch import dtype, int64, nn, std
import torch
import numpy as np
from torch.nn.modules import loss

def weight_init(layers):
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0, std=0.1)
        nn.init.constant_(layer.bias, 0.)

def v_wrap(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)

def push_and_pull(opt, lnet, gnet, done, s_, bs, ba, br, gamma):
    if done:
        v_s_ = np.array([0. for i in range(len(s_))])
    else:
        v_s_ = lnet.forward(v_wrap(s_))[-1].data.numpy()[0,0]
    
    buffer_v_target = []
    for r in br[::-1]:
        v_s_ = r +  v_s_ * gamma
        buffer_v_target.append(v_s_)
    buffer_v_target.reverse()

    loss = lnet.loss_func(
        v_wrap(np.vstack(bs)),
        v_wrap(np.array(ba), dtype=np.int64) if ba[0].dtype == np.int64 else v_wrap(np.vstack(ba)),
        v_wrap(np.array(buffer_v_target)[:,None])
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
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r.mean() * 0.01
    res_queue.put((global_ep_r.value, loss, name, step))
    print(
        name,
        "Ep:", global_ep.value,
        "| Ep_r: %.0f" % global_ep_r.value,
    )