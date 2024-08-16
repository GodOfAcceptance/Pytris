import tetris_env
import pygame
import argparse
from settings import *
from agent import RandomAgent

class Main:
    def __init__(self, player, agent, render_mode, nEpisodes, sfx, DAS, ARR):
        assert player in ["human", "agent"], "unsupported player. Available: human, agent"
        assert nEpisodes > 0, "n_episodes must be greater than 0"

        self.sfx = sfx
        self.player = player
        self.agent = agent
        self.env = tetris_env.TetrisEnv(render_mode=render_mode, DAS=DAS, ARR=ARR)
        self.nEpisodes = nEpisodes
        self.gameIsRunning = False
        self.restart = False
        
        # self.keyHeld = [False] * 8 #Noop, left, right, rotleft, rotright, softdrop, harddrop, hold
        # self.keyHeldFrame = [0] * 8
        self.keyHeld = np.zeros((8,), dtype=bool)
        self.keyHeldFrame = np.zeros((8,), dtype=int)
        
        
    def run(self):
        if self.player == "human":
            self.playHuman()
        elif self.player == "agent":
            self.playAgent(self.nEpisodes)

    
    def playAgent(self, nEpisodes):
        assert self.player == "agent"
        assert self.agent is not None
        self.gameIsRunning = True
        
        for episode in range(nEpisodes):
            if not self.gameIsRunning:
                break;
            
            done, info = self.env.reset()
            
            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        done = True
                        self.gameIsRunning = False
                        break;
                action = self.agent.getAction()
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                if terminated or truncated:
                    done = True
                    break;
        
        self.gameIsRunning = False
        self.env.close()
         

    def playHuman(self):
        self.env.reset()
        self.gameIsRunning = True
        
        while self.gameIsRunning:
            if self.restart:
                self.gameIsRunning = False
                break;
            self.events()
            self.updateKeyHeldFrame()
            
            action = self.getAction()
            obs, reward, terminated, truncated, info = self.env.step(action)
  
            
            if self.sfx and info["locked"]:
                lockSFX.play()
                
            if terminated or truncated:
                self.gameIsRunning = False
                break;
            
        if self.restart:
            self.restart = False
            self.playHuman()
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
        if self.keyHeld[3] and self.keyHeldFrame[3] == 1: #left
            rotation = -1
        elif self.keyHeld[4] and self.keyHeldFrame[4] == 1: #righ
            rotation = 1 
        return rotation
    
    
    def getDrop(self):
        drop = 0
        if self.keyHeld[5]:
            drop = 1
        if self.keyHeld[6] and self.keyHeldFrame[6] == 1:
            drop = 2
        return drop
    
    
    def getAction(self):
        res = np.zeros((4,), dtype=np.int8)
        res[0] = self.getDirection()
        res[1] = self.getRotation()
        res[2] = self.getDrop()
        if self.keyHeld[7] and self.keyHeldFrame[7] == 1:
            res[3] = True
        return res
    
            
        
    
if __name__ == '__main__':
    #main = Main(player="human", render_mode='human', sfx=True)
    #main = Main(player="agent", agent=RandomAgent(), render_mode='human', sfx=True)
    #main.run()
    parser = argparse.ArgumentParser(description='Example')
    parser.add_argument('--player', type=str, default='human', help='human or agent', required=True)
    parser.add_argument('--agent', type=str, default='random', help='agent class', required=False)
    parser.add_argument('--render_mode', type=str, default='human', help='human or rgb_array', required=True)
    parser.add_argument('--sfx', type=bool, default=False, help='sound effects', required=False)
    parser.add_argument('--DAS', type=int, default=8, help='delayed auto shift in frames', required=False)
    parser.add_argument('--ARR', type=int, default=1, help='auto repeat rate in frames', required=False)
    parser.add_argument('--n_episodes', type=int, default=5, help='number of episodes', required=False)
    
    args = parser.parse_args()
    
    if not args.player in ["human", "agent"]:
        print("unsupported player. Available: human, agent")
        exit()
        
    if args.render_mode not in ["human", "rgb_array"]:
        print("unsupported render_mode. Available: human, rgb_array")
        exit()
        
    if args.player == "agent" and args.agent is None:
        print("agent class is required")
        exit()
        
    if args.render_mode == "rgb_array" and args.sfx:
        print("sound effects are not supported in rgb_array mode")
        exit()
    
    if args.player == "human" and args.render_mode == "rgb_array":
        print("human player is not supported in rgb_array mode")
        exit()
    
    if args.DAS < 0 or args.ARR < 0:
        print("DAS and ARR values must be positive")
        exit()
        
    if args.n_episodes < 1:
        print("n_episodes must be greater than 0")
        exit()
        

    agent = None
    
    match args.agent:
        case "random":
            agent = RandomAgent()
        case _:
            exit()
    
    
    main = Main(player=args.player, 
                agent=agent, 
                render_mode=args.render_mode,
                nEpisodes=args.n_episodes, 
                sfx=args.sfx, 
                DAS=args.DAS, 
                ARR=args.ARR,
                )
    main.run()
    
    