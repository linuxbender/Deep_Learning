# Based on:
# https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
# https://arxiv.org/pdf/1509.02971.pdf

import numpy as np

class OUNoise:

    def __init__(self, action_dimension, mu=None, theta=0.15, sigma=0.2):        
        self.action_dimension = action_dimension
        self.mu = mu if mu is not None else np.zeros(self.action_dimension)
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu*2
        self.reset()

    def reset(self):
        self.state = self.mu

    def sample(self):
        # debugger
        # import pdb; pdb.set_trace()

        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

if __name__ == '__main__':
    ou = OUNoise(3)
    states = []
    for i in range(1000):
        states.append(ou.sample())

    import matplotlib.pyplot as plt
    plt.plot(states)
    plt.show()
