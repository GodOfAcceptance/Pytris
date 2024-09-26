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
    

class PathFinder:
    """
    Path finding agent.
    States are possible positions and rotations of the current piece.
    The agent finds the sequence of actions to place the current piece
    at a desired spot. The agent evaluates each spot using heuristics like
    bumpiness.

    When implemented, it should show this behavior:
    1. read current state
    2. find the best state with current piece(s).
    3. find the path (action sequence) to that state
    4. check
    5. wait for the next piece.
    """
    def __init__(self):
        pass


    def getAllPossibleStates(self, board):
        """
        State refers to the placement of piece.
        """
        states = []
        r, c = board.shape
        for i in range(c):
            pass