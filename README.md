A Tetris Environment that inherits gymnasium environment.
This environment implements all modern tetris features: DAS, ARR, SRS, 7Bag randomizer.
The player makes an action through a controller, which receives binary input for each corresponding key.
There are 7 keys: left, right, rotate left, rotate right, soft drop, hard drop, hold.

You can play the game by yourself by running "python main.py". Add --sfx to enable sound effects.
Currently, no reinforcement learning agent is available due to my skill issue.
However, you can run a random agent or keep-dropping agent: python main.py --player "agent" --agent "random" (or "KD")

Action Space: MultiBinary(7)

