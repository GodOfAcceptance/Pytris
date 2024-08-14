import tetris_env

env = tetris_env.TetrisEnv(render_mode=None)
class RandomAgent:
    def __init__(self):
        pass
    
    def getAction(self, obs):
        return env.action_space.sample()
    
    