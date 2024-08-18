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
        
        self.keyHeld = np.zeros((7,), dtype=bool)
        
        
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
            
            obs, info = self.env.reset()
            done = False
            
            while not done:
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        done = True
                        self.gameIsRunning = False
                        break;
                    
                action = self.agent.predict(obs)[0]
                print(action)
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
            obs, reward, terminated, truncated, info = self.env.step(self.keyHeld)
  
            
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
                    self.keyHeld[0] = True
                if event.key == pygame.K_RIGHT:
                    if self.sfx: tapSFX.play()
                    self.keyHeld[1] = True
                if event.key == pygame.K_z:
                    self.keyHeld[2] = True
                if event.key == pygame.K_x:
                    self.keyHeld[3] = True
                if event.key == pygame.K_DOWN:
                    self.keyHeld[4] = True
                if event.key == pygame.K_SPACE:
                    self.keyHeld[5] = True
                if event.key == pygame.K_r:
                    self.restart = True
                if event.key == pygame.K_c:
                    self.keyHeld[6] = True
                    
            
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    self.keyHeld[0] = False
                if event.key == pygame.K_RIGHT:
                    self.keyHeld[1] = False
                if event.key == pygame.K_z:
                    self.keyHeld[2] = False
                if event.key == pygame.K_x:
                    self.keyHeld[3] = False
                if event.key == pygame.K_DOWN:
                    self.keyHeld[4] = False
                if event.key == pygame.K_SPACE:
                    self.keyHeld[5] = False
                if event.key == pygame.K_c:
                    self.keyHeld[6] = False
                    
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example')
    parser.add_argument('--player', type=str, default='human', help='human or agent', required=False)
    parser.add_argument('--agent', type=str, default=None, help='agent class', required=False)
    parser.add_argument('--render_mode', type=str, default='human', help='human or rgb_array', required=False)
    parser.add_argument('--sfx', action='store_true', help='sound effects', required=False)
    parser.add_argument('--DAS', type=int, default=8, help='delayed auto shift in frames', required=False)
    parser.add_argument('--ARR', type=int, default=1, help='auto repeat rate in frames', required=False)
    parser.add_argument('--n_episodes', type=int, default=5, help='number of episodes', required=False)
    parser.add_argument('--train', action='store_true', help='train agent', required=False)
    
    args = parser.parse_args()
    if args.train:
        from stable_baselines3 import DDPG
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.vec_env import SubprocVecEnv
        from gymnasium.wrappers import TimeLimit
        from stable_baselines3.common.env_checker import check_env
        from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

        # Parallel environments
        check_env(tetris_env.TetrisEnv())
        env = tetris_env.TetrisEnv(render_mode="rgb_array")
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

        model = DDPG("MultiInputPolicy", env, verbose=1,  action_noise=action_noise)
        model.learn(total_timesteps=100000, log_interval=10)
        model.save("ddpg")
            
    else:
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
        if args.agent is not None:
            match args.agent:
                case "random":
                    agent = RandomAgent()
                case "ddpg":
                    from stable_baselines3 import DDPG
                    agent = DDPG.load("ddpg")
                case "KD":
                    from agent import KeepDropAgent
                    agent = KeepDropAgent()
                case _:
                    print("unsupported agent")
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
    
    