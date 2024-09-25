import numpy as np
class Controller:
    def __init__(self, DAS, ARR):
        self.DAS = DAS
        self.ARR = ARR
        self.reset()

    
    def reset(self):
        self.keyHeld = np.zeros((7,), dtype=bool)
        self.keyHeldFrames = np.zeros((7,), dtype=int)
        self.das = 0
        self.arr = 0

    
    def update(self, input):
        prevKeyHeld = self.keyHeld
        self.keyHeld = input.copy()
        self.keyHeldFrames = np.where(self.keyHeld, self.keyHeldFrames + 1, 0)

        if self.justReleasedLeftOrRight(self.keyHeld, prevKeyHeld):
            self.das = 0
            self.arr = self.ARR


        if self.justPressedLeftOrRight():
            #reset das/arr
            self.das = 1
            self.arr = self.ARR
            return self.getXDirection()


        if self.isPressingLeftOrRight():
            if self.das >= self.DAS:
                if self.arr >= self.ARR:
                    self.arr = 0
                    return self.getXDirection()
                else:
                    self.arr += 1
            else:
                self.das += 1
        else:
            self.das = 0
            self.arr = 0

        print(self.das)
        return 0    


    def getXDirection(self):
        assert self.keyHeld[0] or self.keyHeld[1]

        if not self.keyHeld[0]:
            return 1
        elif not self.keyHeld[1]:
            return -1
        else:
            if self.keyHeldFrames[0] >= self.keyHeldFrames[1]:
                return 1
            else:
                return -1

    
    def justReleasedLeftOrRight(self, currKeyHeld, prevKeyHeld):
        return (not currKeyHeld[0] and prevKeyHeld[0]) or (not currKeyHeld[1] and prevKeyHeld[1])


    def justPressedLeftOrRight(self):
        return (self.keyHeld[0] and self.keyHeldFrames[0] == 1) or (self.keyHeld[1] and self.keyHeldFrames[1] == 1)
    

    def isPressingLeftOrRight(self):
        return self.keyHeld[0] or self.keyHeld[1]