This is a game of tetris with proper DAS, ARR, SRS support. 
Many implementations of modern tetris does not handle inputs well, but mine does handle it correctly.

settings.py -- settings like screen layout, DAS, ARR, tetrominoes, etc.
tetris_env.py -- a tetris game state that implements gymnasium.Env. Could be improved and used for future RL training.
main.py -- main application from which you can play the game.
agent.py -- contains a random action agent.

How to play: make sure the instance of tetris environment under if __name__ = "__main__" have player="human" in order to play by yourself.
Then, call python main.py on terminal.
