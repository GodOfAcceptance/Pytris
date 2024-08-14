import tetris_env

env = tetris_env.TetrisEnv(render_mode=None)
class RandomAgent:
    mapping = {0: 0, 1: -1, 2: 1}
    def __init__(self):
        pass
    
    def getAction(self):
        return env.action_space.sample()
    
