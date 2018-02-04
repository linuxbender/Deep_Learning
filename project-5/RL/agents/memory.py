import random
from collections import namedtuple

Experience = namedtuple("Experience",field_names=["state", "action", "reward", "next_state", "done"])

class Memory:    

    def __init__(self, size=1000):        
        self.size = size  # maximum size of buffer
        self.memory = []  # internal memory (list)
        self.idx = 0  # current index into circular buffer
    
    def add(self, state, action, reward, next_state, done):
        e = Experience(state, action, reward, next_state, done)
        if len(self.memory) < self.size:
            self.memory.append(e)
        else:
            self.memory[self.idx] = e
            self.idx = (self.idx + 1) % self.size
    
    def sample(self, batch_size=64):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

if __name__ == '__main__':
    mem = Memory(10)
    for i in range(15):
        mem.add(i, i % 2, i % 3 - 1, i + 1, i % 4)
    for i, e in enumerate(mem.memory):
        print(i, e)
    batch = mem.sample(5)
    print("Random batch: size = {}".format(len(batch)))  # maximum size if full
    for e in batch:
        print(e)
