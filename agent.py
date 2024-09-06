import tetris_env
from settings import *
import numpy as np
import itertools

class InvalidActionException(Exception):
    pass

class InvalidStateException(Exception):
    pass

env = tetris_env.TetrisEnv( render_mode="human")
class RandomAgent:
    def __init__(self):
        pass
    
    def predict(self, obs):
        return [env.action_space.sample()]
    
class KeepDropAgent:
    def __init__(self):
        pass
    
    def predict(self, obs):
        return [[0,0,0,0,1,0,0]]
            
