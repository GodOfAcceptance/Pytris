This is a game of tetris with proper DAS, ARR, SRS support. 
Many implementations of modern tetris does not handle inputs well, but mine does handle it correctly.

settings.py -- settings like screen layout, DAS, ARR, tetrominoes, etc.
tetris_env.py -- a tetris game state that implements gymnasium.Env. Could be improved and used for future RL training.
main.py -- main application from which you can play the game.
agent.py -- contains a random action agent.

How to play: 
Type "python main.py" into command line. Specify required arguments: player and render mode. 
Currently, you can only play as yourself or a random action agent.


