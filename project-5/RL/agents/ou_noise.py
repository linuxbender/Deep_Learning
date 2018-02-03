# Based on:
# https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py

import numpy as np

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, action_dimension, mu=None, theta=1.5, sigma=4):
        """Initialize parameters and noise process."""
        self.action_dimension = action_dimension
        self.mu = mu if mu is not None else np.zeros(self.action_dimension)
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""        
        self.state = self.mu

    def sample(self):
        """Update internal state and return it as a noise sample."""
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
