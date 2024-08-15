import tetris_env
from collections import deque
import random
from collections import namedtuple

env = tetris_env.TetrisEnv(render_mode=None)
class RandomAgent:
    mapping = {0: 0, 1: -1, 2: 1}
    def __init__(self):
        pass
    
    def getAction(self):
        return env.action_space.sample()
    


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
class ReplayMemory:
    def __init__(self, capacity):
        self.mem = deque([], capacity)
    
    
    def push(self, *args):
        self.mem.append(Transition(*args))
        
    
    def sample(self, batch_size):
        return random.sample(self.mem, batch_size)
    
    
    def __len__(self):
        return len(self.mem)
    
    