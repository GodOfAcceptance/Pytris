import numpy as np
from enum import Enum

class Controller:
    LEFT = 0
    RIGHT = 1
    RLEFT = 2
    RRIGHT = 3
    SOFT = 4
    HARD = 5
    HOLD = 6
    def __init__(self, DAS, ARR):
        self.reset()

    
    def reset(self):
        self.keyHeld = np.zeros((7,), dtype=bool)
        self.keyHeldFrames = np.zeros((7,), dtype=int)


    def update(self, input):
        self.keyHeld = input.copy() #is this necessary?
        self.keyHeldFrames = np.where(input, self.keyHeldFrames + 1, 0)

    
    def isPressed(self, key):
        return self.keyHeld[key]
    

    

