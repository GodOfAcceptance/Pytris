import numpy as np
class Controller:
    def __init__(self):
        self.keyHeld = [False] * 7
        self.keyHeldFrame = [0] * 7
        
        
    def updateKeyHeldFramesForAgent(self, action):
        # for i in range(len(self.keyHeld)):
        #     self.keyHeld[i] = action[i]
        self.keyHeld = action
        self.updateKeyHeldFrame()
                        
    
    def updateKeyHeldFrame(self):
        for i in range(len(self.keyHeld)):
            if self.keyHeld[i]:
                self.keyHeldFrame[i] += 1
            else:
                self.keyHeldFrame[i] = 0
                
                
    def getDirection(self):
        direction = 0
        if self.keyHeld[0] and self.keyHeld[1]:
            direction = 1 if self.keyHeldFrame[0] < self.keyHeldFrame[1] else 2
        elif self.keyHeld[0] and not self.keyHeld[1]:
            direction = 1
        elif self.keyHeld[1] and not self.keyHeld[0]:
            direction = 2
            
        return direction
    
    
    def getRotation(self):
        rotation = 0 #counter clockwise
        if self.keyHeld[2] and self.keyHeldFrame[2] == 1: #left
            rotation = 1
        elif self.keyHeld[3] and self.keyHeldFrame[3] == 1: #right
            rotation = 2 
        return rotation
    
    
    def getDrop(self):
        drop = 0
        if self.keyHeld[4]:
            drop = 1
        if self.keyHeld[5] and self.keyHeldFrame[5] == 1:
            drop = 2
        return drop
    
    
    def getAction(self):
        res = np.zeros((4,), dtype=np.int8)
        res[0] = self.getDirection()
        res[1] = self.getRotation()
        res[2] = self.getDrop()
        if self.keyHeld[6] and self.keyHeldFrame[6] == 1:
            res[3] = True
        return self.convert_action(res)
    
    
    def convert_action(self, rawAction):
        """
        Converts the raw action [0,1,2] into [-1,0,1]
        """
        trueAction = [0,0,0,0]
        trueAction[0] = 0 if rawAction[0] == 0 else -1 if rawAction[0] == 1 else 1
        trueAction[1] = 0 if rawAction[1] == 0 else -1 if rawAction[1] == 1 else 1
        trueAction[2] = rawAction[2]
        trueAction[3] = rawAction[3]
        return trueAction