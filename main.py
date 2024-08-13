import tetris_env
import pygame
import time
from settings import *

class Main:
    def __init__(self, training=False, agent=None, render_mode=None, sfx=False):
        assert not (training and sfx), "Don't enable sfx while training!"
        self.sfx = sfx
        self.isTraining = training
        self.env = tetris_env.TetrisEnv(render_mode=render_mode)
        self.gameIsRunning = False
        self.restart = False
        
        self.keyHeld = [False] * 8 #Noop, left, right, rotleft, rotright, softdrop, harddrop, hold
        self.keyHeldFrame = [0] * 8

    def run(self):
        self.play()
            
            
    def train(self):
        pass
    
         

    def play(self):
        self.env.reset()
        self.gameIsRunning = True
        
        while self.gameIsRunning:
            if self.restart:
                self.gameIsRunning = False
                break;
            self.events()
            self.updateKeyHeldFrame()
            action = self.getAction()
            
            done, info = self.env.step(action)
            
            if self.sfx and info["locked"]:
                lockSFX.play()
                
            if done:
                self.isRunning = False
                break;
            
        if self.restart:
            self.restart = False
            self.play()
        self.env.close()
        

    def events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.gameIsRunning = False
                break;
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    if self.sfx: tapSFX.play()
                    self.keyHeld[1] = True
                if event.key == pygame.K_RIGHT:
                    if self.sfx: tapSFX.play()
                    self.keyHeld[2] = True
                if event.key == pygame.K_z:
                    self.keyHeld[3] = True
                if event.key == pygame.K_x:
                    self.keyHeld[4] = True
                if event.key == pygame.K_DOWN:
                    self.keyHeld[5] = True
                if event.key == pygame.K_SPACE:
                    self.keyHeld[6] = True
                if event.key == pygame.K_r:
                    self.restart = True
                if event.key == pygame.K_c:
                    self.keyHeld[7] = True
                    
            
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    self.keyHeld[1] = False
                if event.key == pygame.K_RIGHT:
                    self.keyHeld[2] = False
                if event.key == pygame.K_z:
                    self.keyHeld[3] = False
                if event.key == pygame.K_x:
                    self.keyHeld[4] = False
                if event.key == pygame.K_DOWN:
                    self.keyHeld[5] = False
                if event.key == pygame.K_SPACE:
                    self.keyHeld[6] = False
                if event.key == pygame.K_c:
                    self.keyHeld[7] = False
                    
                    
    
    def updateKeyHeldFrame(self):
        for i in range(len(self.keyHeld)):
            if self.keyHeld[i]:
                self.keyHeldFrame[i] += 1
            else:
                self.keyHeldFrame[i] = 0
    
    
    def getDirection(self):
        direction = 0
        if self.keyHeld[1] and self.keyHeld[2]:
            direction = -1 if self.keyHeldFrame[1] < self.keyHeldFrame[2] else 1
        elif self.keyHeld[1] and not self.keyHeld[2]:
            direction = -1
        elif self.keyHeld[2] and not self.keyHeld[1]:
            direction = 1
            
        return direction
    
    
    def getRotation(self):
        rotation = 0 #counter clockwise
        if self.keyHeld[3] and self.keyHeldFrame[3] == 1:
            rotation = 1
        elif self.keyHeld[4] and self.keyHeldFrame[4] == 1:
            rotation = -1 
        return rotation
    
    
    def getDrop(self):
        drop = 0
        if self.keyHeld[5]:
            drop = 1
        if self.keyHeld[6] and self.keyHeldFrame[6] == 1:
            drop = 2
        return drop
    
    
    def getAction(self):
        NOOP = True
        direction = self.getDirection()
        rotation = self.getRotation()
        drop = self.getDrop()
        hold = False
        if self.keyHeld[7] and self.keyHeldFrame[7] == 1:
            hold = True
        if direction != 0 or rotation != 0 or drop != 0:
            NOOP = False
        return [NOOP, direction, rotation, drop, hold]
    
            
        
    
if __name__ == '__main__':
    main = Main(training=False, render_mode='human', sfx=True)
    main.run()
    
    