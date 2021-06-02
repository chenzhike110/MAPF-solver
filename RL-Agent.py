from simulator import Simulator
from a3c.discrete_A3C import net, Worker

if __name__ == "__main__":
    model = net(3)
    env = Simulator((601,601,3),8)
    done = False
    state = env.reset()
    env.show()
    while not done:
        action = model.choose_action(state)
        print(action)
        reward, states, done, _ = env.step(action)
        env.show()
