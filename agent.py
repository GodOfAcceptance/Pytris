import tetris_env

env = tetris_env.TetrisEnv(render_mode="human")
class RandomAgent:
    def __init__(self):
        pass
    
    def predict(self, obs):
        return [env.action_space.sample()]
