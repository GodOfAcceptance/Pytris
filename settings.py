import pygame.font
import pygame.mixer
import numpy as np
pygame.font.init()
pygame.mixer.init()



## SCREEN
#Board layout
# COLUMNS = 12 # 10 cols + 2 walls
# ROWS = 23 # 20 rows + 2 hidden + 1 wall
# CELL_SIZE = 30
COLUMNS = 10
HIDDEN_ROWS = 2
ROWS = 20 + HIDDEN_ROWS
CELL_SIZE = 30

BOARD_SURFACE_WIDTH = COLUMNS * CELL_SIZE
BOARD_SURFACE_HEIGHT = (ROWS - HIDDEN_ROWS)*CELL_SIZE

#Preview layout
NUM_PREVIEW = 5
PREVIEW_WIDTH = 6 * CELL_SIZE
PREVIEW_HEIGHT = (NUM_PREVIEW*3+1) * CELL_SIZE

#Info layout
STAT_WIDTH = BOARD_SURFACE_WIDTH
STAT_HEIGHT = 4 * CELL_SIZE

#Hold layout
HOLD_WIDTH = 5*CELL_SIZE
HOLD_HEIGHT = 5*CELL_SIZE

#Window
PADDING = 20
WINDOW_WIDTH = PADDING + HOLD_WIDTH + BOARD_SURFACE_WIDTH  + PREVIEW_WIDTH + PADDING
WINDOW_HEIGHT = PADDING + BOARD_SURFACE_HEIGHT + STAT_HEIGHT + PADDING

FONT = pygame.font.SysFont("arial", 18)
GRAY = (128,128,128)
BORDER_GRAY = (60, 60, 60)


## GAME MECHANICS
#Numbers are in terms of frames
DROP_INTERVAL = 60
SOFT_DROP_INTERVAL = 1
LOCK_DELAY = 60
FORCE_LOCK_DELAY = 500


# DAS = 6
# ARR = 0 #if 0, then teleport

SPAWN_X = COLUMNS // 2 - 1
SPAWN_Y = 0

NEW_LINE = np.zeros((COLUMNS,), dtype=int)
NEW_LINE[0] = 9
NEW_LINE[COLUMNS-1] = 9



## Sound Effects
tapSFX = pygame.mixer.Sound("sfx/tap.wav")
tapSFX.set_volume(0.5)
lockSFX = pygame.mixer.Sound("sfx/lock.wav")
holdSFX = pygame.mixer.Sound("sfx/hold.wav")


## Tetrominoes
#All tetrominoes except I and O are represented as 3x3 matrices, flattened to 1D array.
TETROMINO_default = np.empty((7,), dtype=object)

TETROMINO_default[0] = [0,0,0,0]
TETROMINO_default[0] +=[1,1,1,1]
TETROMINO_default[0] +=[0,0,0,0]
TETROMINO_default[0] +=[0,0,0,0]
TETROMINO_default[0] = np.reshape(TETROMINO_default[0], (4,4))

TETROMINO_default[1] =  [2,0,0]
TETROMINO_default[1] += [2,2,2]
TETROMINO_default[1] += [0,0,0]

TETROMINO_default[2] =  [0,0,3]
TETROMINO_default[2] += [3,3,3]
TETROMINO_default[2] += [0,0,0]


TETROMINO_default[3] =  [0,4,4]
TETROMINO_default[3] += [0,4,4]
TETROMINO_default[3] += [0,0,0]

TETROMINO_default[4] =  [0,5,5]
TETROMINO_default[4] += [5,5,0]
TETROMINO_default[4] += [0,0,0]

TETROMINO_default[5] =  [0,6,0]
TETROMINO_default[5] += [6,6,6]
TETROMINO_default[5] += [0,0,0]

TETROMINO_default[6] =  [7,7,0]
TETROMINO_default[6] += [0,7,7]
TETROMINO_default[6] += [0,0,0]

for i in range(1,7):
    TETROMINO_default[i] = np.reshape(TETROMINO_default[i], (3,3))
    
    
TETROMINO = {}
for i in range(7):
    rotatable = [] #counter clockwise for each index
    for j in range(4):
        if i == 3:
            rotatable.append(TETROMINO_default[i])
        else:
            rotatable.append(np.rot90(TETROMINO_default[i], j))
    TETROMINO[i] = rotatable
    


TETROMINO_COLORS = {
    1: (15,155,215),
    2: (33,65,198),
    3: (227,91,2),
    4: (227,159,2),
    5: (89,177,1),
    6: (175,41,138),
    7: (215,15,55),   
}


#key = (initial rotation, attempted rotation). 0 is spawn state, 1 is counter-clockwise(left), 2 is flip, 3 is right.
#value = five (x,y) coordinate translation. Note that the array is indexed row-first
#so, piece[py + y][px + x] is the right way to use this.

WALL_KICK_DATA = {}
WALL_KICK_DATA[(0,3)] = [(0,0),(-1,0),(-1,1),(0,-2),(-1,-2)]
WALL_KICK_DATA[(3,0)] = [(0,0),(1,0),(1,-1),(0,2),(1,2)]
WALL_KICK_DATA[(3,2)] = [(0,0),(1,0),(1,-1),(0,2),(1,2)]
WALL_KICK_DATA[(2,3)] = [(0,0),(-1,0),(-1,1),(0,-2),(-1,-2)]
WALL_KICK_DATA[(2,1)] = [(0,0),(1,0),(1,1),(0,-2),(1,-2)]
WALL_KICK_DATA[(1,2)] = [(0,0),(-1,0),(-1,-1),(0,2),(-1,2)]
WALL_KICK_DATA[(1,0)] = [(0,0),(-1,0),(-1,-1),(0,2),(-1,2)]
WALL_KICK_DATA[(0,1)] = [(0,0),(1,0),(1,1),(0,-2),(1,-2)]


WALL_KICK_DATA_I = {}
WALL_KICK_DATA_I[(0,3)] = [(0,0),(-2,0),(1,0),(-2,-1),(1,2)]
WALL_KICK_DATA_I[(3,0)] = [(0,0),(2,0),(-1,0),(2,1),(-1,-2)]
WALL_KICK_DATA_I[(3,2)] = [(0,0),(-1,0),(2,0),(-1,2),(2,-1)]
WALL_KICK_DATA_I[(2,3)] = [(0,0),(1,0),(-2,0),(1,-2),(-2,1)]
WALL_KICK_DATA_I[(2,1)] = [(0,0),(2,0),(-1,0),(2,1),(-1,-2)]
WALL_KICK_DATA_I[(1,2)] = [(0,0),(-2,0),(1,0),(-2,-1),(1,2)]
WALL_KICK_DATA_I[(1,0)] = [(0,0),(1,0),(-2,0),(1,-2),(-2,1)]
WALL_KICK_DATA_I[(0,1)] = [(0,0),(-1,0),(2,0),(-1,2),(2,-1)]




REWARD_MAP = {
    0: 0,
    1: 100,
    2: 300,
    3: 500,
    4: 800,
}

BTB_MULTIPLIER = 1.5