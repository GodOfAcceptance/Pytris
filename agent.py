import tetris_env

env = tetris_env.TetrisEnv(render_mode=None)
class RandomAgent:
    def __init__(self):
        pass
    
    def getAction(self):
        rawAction = env.action_space.sample()
        trueAction = [0,0,0,0]
        trueAction[0] = (2 * rawAction[0] - 3) * (rawAction[0] != 0)
        trueAction[1] = (2 * rawAction[1] - 3) * (rawAction[1] != 0)
        trueAction[2] = rawAction[2]
        trueAction[3] = rawAction[3]
        return trueAction

    
