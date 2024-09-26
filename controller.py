import numpy as np
from enum import Enum
class Key(Enum):
    LEFT = 0
    RIGHT = 1
    RLEFT = 2
    RRIGHT = 3
    SOFT = 4
    HARD = 5
    HOLD = 6


class Controller:
    def __init__(self):
        self.reset()

    
    def reset(self):
        self.keyHeld = np.zeros((7,), dtype=bool)
        self.keyHeldFrames = np.zeros((7,), dtype=int)


    def update(self, input):
        self.keyHeld = input.copy() #is this necessary?
        self.keyHeldFrames = np.where(input, self.keyHeldFrames + 1, 0)

    
    def justPressedLeftOrRight(self):
        left = (self.keyHeld[0] and self.keyHeldFrames[0] == 1)
        right = (self.keyHeld[1] and self.keyHeldFrames[1] == 1)
        return left or right


    def isPressingLeftOrRight(self):
        return self.keyHeld[0] or self.keyHeld[1]
    

    def justPressedKey(self, key: Key):
        return self.keyHeldFrames[key.value] == 1


    def isPressingKey(self, key: Key):
        return self.keyHeldFrames[key.value] >= 1


