import numpy as np
from settings import *
import gymnasium as gym
from gymnasium import spaces
from collections import deque



class TetrisEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}
    """
    ## Description
    
    This environment represents an internal game state of Tetris. Implements
    various modern Tetris features such as:
        - SRS (Super Rotation System)
        - DAS and ARR (Delayed Auto Shift and Auto Repeat Rate)
        - Preview of next pieces
        - Hold piece (will be available in the next version)
    
    
    Action space: MultiDiscrete([3,3,3,2]):
        - Horizontal direction: NOOP[0], LEFT[1], Right[2]
        - Rotation: NOOP[0], LEFT[1], RIGHT[2]
        - Drop type: NOOP[0], Soft[1], hard[2]
        - Hold: NOOP[0], HOLD[1]
    
    Observation space: spaces.Dict(
             {"board": self.board.copy(), 
                "curr": self.curr_piece_type, 
                "rotation": self.rotation, 
                "pos": (self.px, self.py), 
                "hold": self.heldPiece,
                "preview": self.queue.copy()
                })
    
    """
    def __init__(self, render_mode = None, DAS=8, ARR=1):
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        """
        If human-rendering is used, `self.screen` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        super().__init__()
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        
        self.board = None
        self.preview = None
        self.stats = None
        
        self.DAS = DAS
        self.ARR = ARR #if 0, then teleport
        
        
        self.action_space = spaces.MultiDiscrete([3,3,3,2], dtype=int)
    
        self.observation_space = spaces.Dict({
            "board": spaces.Box(0, 9, shape=(ROWS,COLUMNS), dtype=int),
            # "curr": spaces.Discrete(7), 
            # "rotation": spaces.Discrete(4), #0, 1, 2, 3
            # "pos": spaces.Box(low=np.array([1, 0]), high=np.array([COLUMNS - 2, ROWS - 2]), dtype=int),
            # "hold": spaces.Discrete(8,), #7 tetrominoes + none
            # "preview": spaces.Box(0,6, shape=(NUM_PREVIEW,), dtype=int),
        })
    
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.gameOver = False
        self.totalScore = 0
        self.totalSteps = 0
        
        ## Board setup
        self._resetBoard()
        
        ## Piece setup
        self.bag = self._7bagRandomizer()
        self._initializePieceQueue()
        self._spawn()
        self._setGhostCoord()
        self.holdAllowed = True
        self.heldPiece = 7 #none
        
        
        ## Features setup
        self.gravityOn = True
        self.softDropOn = False
        self.startLocking = False
        
        self.previous_dir = 0
        self.das_t = 0
        self.arr_t = 0
        self.lock_t = 0
        self.drop_t = DROP_INTERVAL
        self.soft_t = SOFT_DROP_INTERVAL
        self.force_lock_t = 0



        ## INFO
        self.timeElapsed = 0.0

        if self.render_mode == 'human':
            self.render()
        
        return self._getObs(), {}
    
    
        
    def step(self, action):
        """
        action = [Horizontal Direction, Rotation, Drop Type, Hold]
        """
        reward = 0
        clearCount = 0
        truncated = self.totalSteps > 5000 and self.totalScore < 100
        terminated = self.gameOver
        info = {"locked": False, "hold": False} #used for vfx in human play mode.
        
        if self.gameOver:
            return self._getObs(), 0, terminated, truncated, info
        
        
        
        self.timeElapsed = round(self.timeElapsed + 1.0 / self.metadata["render_fps"], 2)
        now = int(pygame.time.get_ticks() / 1000.0 * self.metadata["render_fps"]) ##self.reset() calls pygame.init(), so assume it's safe
        trueAction = self.convert_action(action)
        direction = trueAction[0]
        rotation = trueAction[1]
        drop = trueAction[2]
        hold = trueAction[3]
        
        if hold and self.holdAllowed: #hold
            self._swap()
            info["hold"] = True
            observation = self._getObs()
            return observation, reward, terminated, truncated, info
            
            
        if drop == 2: #Hard drop
            self._hard_drop()
            self._lock_piece(self.curr_piece_type, self.rotation, self.px, self.py)
            clearCount = self._clearLines()
            info["locked"] = True
            self._spawn()
             
        else:
            if drop == 1:
                self.gravityOn = False
                self.softDropOn = True
            elif drop == 0:
                self.gravityOn = True
                self.softDropOn = False
            else:
                print("unreachable")
                truncated = True
                
            if self.previous_dir != direction or direction == 0:
                self.resetDAS()
                
            if direction != 0:
                self.lock_t = 0.0
                if self.das_t == 0:
                    self.das_t = now
                    self.arr_t = now
                    self.horizontalMove(direction)
                elif (now - self.das_t >= self.DAS) and (now - self.arr_t >= self.ARR):
                    instant = self.ARR == 0
                    self.horizontalMove(direction, instant)
                    self.arr_t = now

            if rotation != 0:
                self.lock_t = 0.0
                if self._doesFit(self.curr_piece_type, (self.rotation - rotation) % 4, self.px, self.py):
                    self.rotation = (self.rotation - rotation) % 4
                else:
                    self._wallKick(self.curr_piece_type, (self.rotation - rotation) % 4)
            
            
            if self.gravityOn:
                self.soft_t = SOFT_DROP_INTERVAL
                if self.drop_t >= DROP_INTERVAL:
                    self._gravity_fall()
                    self.drop_t = 0.0
                else:
                    self.drop_t += 1
            
            if self.softDropOn:
                self.drop_t = 0.0
                if self.soft_t >= SOFT_DROP_INTERVAL:
                    self._softDrop()
                    self.soft_t = 0.0
                else:
                    self.soft_t += 1
            
            self.startLocking = not self._doesFit(self.curr_piece_type, self.rotation, self.px, self.py + 1)
            if self.startLocking:
                self.force_lock_t += 1
                if self.lock_t >= LOCK_DELAY or self.force_lock_t >= FORCE_LOCK_DELAY:
                    self._lock_piece(self.curr_piece_type, self.rotation, self.px, self.py)
                    clearCount = self._clearLines()
                    info["locked"] = True
                    self._spawn()
                else:
                    self.lock_t += 1

        self._setGhostCoord()
        
        self.totalScore += REWARD_MAP[clearCount] #this is for human display only.
        ## start reward calculation
        reward = 0 #TODO
        
        if self.render_mode == 'human':
            self.render()

        observation = self._getObs()
            
        return observation, reward, terminated, truncated, info
    
    
    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization"
            )
        else:
            return self._render_frame(self.render_mode)
    
            
    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()

    
    
    
    
    
    
    ##################   HELPER FUNCTIONS   ######################
    def _nHoles(self, board):
        """
        Returns the number of holes in the board.
        A hole is defined as an empty cell with at least one filled cell above it.
        """
        rows, cols = board.shape
        holes = 0

        for col in range(cols):
            filled = False
            for row in range(rows):
                if board[row][col] != 0:
                    filled = True
                elif filled:
                    # There's a filled cell above and this cell is empty, so it's a hole.
                    holes += 1
        return holes
    
    
    def _nPits(self, board):
        """
        Returns the number of pits in the board.
        A pit is defined as an empty column of height >= 2, surrounded by filled columns on the sides.
        """
        #Get the colum height of left and right. Then, check if the middle column is a pit.
        count = 0
        for col in range(1, COLUMNS-1):
            left = self._columnHeight(board, col-1)
            right = self._columnHeight(board, col+1)
            middle = self._columnHeight(board, col)
            if middle >= 2 and left >= middle and right >= middle:
                count += 1
                
        return count
    

    def _columnHeight(self, board, col):
        """
        Returns the height of the column.
        Requires: 0 <= col < COLUMNS.
        
        col = 0 and col = COLUMNS-1 always return ROWS-1.
        """
        for row in range(ROWS-1):
            if board[row][col] != 0:
                return ROWS-1 - row #don't include the bottom wall.
        return 0
                    
        
        
    def convert_action(self, rawAction):
        """
        Converts the raw action [0,1,2] into [-1,0,1]
        """
        trueAction = [0,0,0]
        trueAction[0] = 0 if rawAction[0] == 0 else 1 if rawAction[0] == -1 else 1
        trueAction[1] = 0 if rawAction[1] == 0 else 1 if rawAction[1] == -1 else 1
        trueAction[2] = rawAction[2]
        trueAction[3] = rawAction[3]
        return trueAction
         
    def _resetBoard(self):
        self.board = np.zeros((ROWS,COLUMNS), dtype=int)
        self.board[:,0] = 9
        self.board[:,COLUMNS-1] = 9
        self.board[ROWS-1,:] = 9
        
        
    def _getObs(self):
        obs =  {"board": self.board.copy(), 
                # "curr": self.curr_piece_type, 
                # "rotation": self.rotation, 
                # "pos": np.array([self.px, self.py]), 
                # "hold": self.heldPiece,
                # "preview": np.array(self.queue)
                }
        return obs

    
    def _initializePieceQueue(self):
        self.queue = deque()
        for i in range(NUM_PREVIEW):
            self.queue.append(next(self.bag))
    
    
    def _7bagRandomizer(self):
        """
        A generator that yields a new tetromino based on 7-bag system
        """
        pieces = np.arange(7)
        while True:
            np.random.shuffle(pieces)
            for piece in pieces:
                yield piece
        
    
    def _clearLines(self):
        """
        Clears rows that are completed.
        Returns the number of lines cleared.
        """
        
        flag = np.all(self.board[:ROWS-1] != 0, axis=1)
        rowsToClear = np.where(flag)[0]
        clearCount = len(rowsToClear)
        if np.any(flag):
            bottom = np.max(rowsToClear)
        else:
            bottom = -1
    
        if clearCount > 0:
            self._pullBoardDown(clearCount, bottom)
        
        
        assert self.board.shape == (ROWS, COLUMNS)
        return clearCount
            
    
    def _pullBoardDown(self, count, start):
        clipped = self.board[:start-count+1].copy()
        fullClip = clipped
        for _ in range(count):
            fullClip = np.vstack([NEW_LINE, fullClip])
        self.board[start-count+1:start+1] = NEW_LINE
        self.board[:start+1] = fullClip
    
    
    def _setGhostCoord(self):
        self.ghostX = self.px
        self.ghostY = self.py
        while self._doesFit(self.curr_piece_type, self.rotation, self.ghostX, self.ghostY + 1):
            self.ghostY += 1
        
        
    def resetDAS(self):
        self.das_t = 0.0
        self.arr_t = 0.0
        
        
    def horizontalMove(self, direction, instant=False):
        if instant:
            while self._doesFit(self.curr_piece_type, self.rotation, self.px+direction, self.py):
                self.px += direction
                self.previous_dir = direction
        else:
            if self._doesFit(self.curr_piece_type, self.rotation, self.px + direction, self.py):
                self.px += direction
                self.previous_dir = direction
    
    
    def _wallKick(self, piece, attempted_rotation):
        assert self.rotation >= 0 and self.rotation <= 3, 'Illegal rotation value. This should not happen'
        assert attempted_rotation >= 0 and attempted_rotation <= 3, 'Illegal rotation value. This should not happen'

        if(piece == 0): #I-mino
            translation_vectors = WALL_KICK_DATA_I[(self.rotation, attempted_rotation)]
            for vector in translation_vectors:
                if(self._doesFit(piece, attempted_rotation, self.px + vector[0], self.py - vector[1])):
                    self.px += vector[0]
                    self.py -= vector[1]
                    self.rotation = attempted_rotation
                    break;
        
        elif(piece in [1,2,4,5,6]):
            translation_vectors = WALL_KICK_DATA[(self.rotation, attempted_rotation)]
            for vector in translation_vectors:
                if(self._doesFit(piece, attempted_rotation, self.px + vector[0], self.py - vector[1])):
                    self.px += vector[0]
                    self.py -= vector[1]
                    self.rotation = attempted_rotation
                    break;
                

    def _gravity_fall(self):
        shouldLock = True
        if self._doesFit(self.curr_piece_type, self.rotation, self.px, self.py + 1):
            self.py += 1
            shouldLock = False
        
        return shouldLock
            

    
    def _hard_drop(self):
        while self._doesFit(self.curr_piece_type, self.rotation, self.px, self.py + 1):
            self.py += 1
     
    
    def _softDrop(self):
        shouldLock = True
        if self._doesFit(self.curr_piece_type, self.rotation, self.px, self.py + 1):
                self.py += 1
                shouldLock = False
            
        return shouldLock

    
    
    def _lock_piece(self, piece_type, rotation, x, y):
        """
        Locks the piece on (x,y).
        """
        piece_array = self._rotate(piece_type, rotation)
        for px in range(piece_array.shape[1]):
            for py in range(piece_array.shape[0]):
                if(piece_array[py,px] != 0):
                    self.board[y+py][x+px] = piece_type + 1
        

    

    def _swap(self):
        """
        Swaps the current piece with the held piece.
        Basically the same as spawn but does not modify the bag.
        """
        if self.heldPiece == 7:
            self.heldPiece = self.curr_piece_type
            self.curr_piece_type = self.queue.popleft()
            self.queue.append(next(self.bag))
        else:
            temp = self.curr_piece_type
            self.curr_piece_type = self.heldPiece
            self.heldPiece = temp
        
        self.px = SPAWN_X
        self.py = SPAWN_Y
        self.rotation = 0
        
        self.drop_t = DROP_INTERVAL
        self.soft_t = SOFT_DROP_INTERVAL
        self.lock_t = 0.0
        
        self.holdAllowed = False
        
        if not self._doesFit(self.curr_piece_type, self.rotation, self.px, self.py):
            self.gameOver = True
    
                            
    def _spawn(self):
        """
        Requires self.queue to be initialized using _initializePieceQueue()
        """
        assert self.queue is not None
        self.curr_piece_type = self.queue.popleft()
        self.rotation = 0
        self.px = SPAWN_X
        self.py = SPAWN_Y
 
        self.drop_t = DROP_INTERVAL
        self.soft_t = SOFT_DROP_INTERVAL
        self.lock_t = 0
        self.force_lock_t = 0
        self.queue.append(next(self.bag))
        
        self.holdAllowed = True
    
                
        if not self._doesFit(self.curr_piece_type, self.rotation, self.px, self.py):
            self.gameOver = True

        
        
    def _rotate(self, piece_type, rotation):
        """
        Given a type of piece, returns a 2D array representing the piece after rotation.
        """
        assert rotation >= 0 and rotation <= 3
        # return np.rot90(TETROMINO[piece_type], k=rotation)
        return TETROMINO[piece_type][rotation]
    

    def _doesFit(self, piece_type, rotation, x, y):
        """
        Returns True if the piece with the given rotation fits into the board at coordinates (x,y).
        The coordinate of the piece is the top-left corner of the piece.
        """
        piece_array = self._rotate(piece_type, rotation)
        for px in range(piece_array.shape[1]):
            for py in range(piece_array.shape[0]):
                if(px + x >= 0 and px + x < COLUMNS):
                    if(py + y >= 0 and py + y < ROWS):
                        if(piece_array[py,px] != 0 and self.board[y+py][x+px] != 0):
                            return False
                        
        return True
    
    

    def _render_frame(self, mode):
        assert mode in self.metadata["render_modes"]
        if self.screen is None and mode == 'human':
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
            self.renderTimeCooldown = 0.0
            self.timeLabel = FONT.render(str(format(0.0, ".2f")), 5, GRAY)
        
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        assert self.screen is not None
        
        self.canvas = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.board_surface = None
        self.preview_surface = None
        self.hold_surface = None
        self.stat_surface = None
        #surface representation of the screen. Everything is rendered here, and displayed
        #only if the render mode is 'human'.
        
        
        self._render_board(self.canvas)
        self._render_preview(self.canvas)
        self._render_hold(self.canvas)
        self._render_stats(self.canvas)
        
        if mode == 'human':
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            self.screen.fill(0)
            self.screen.blit(self.canvas, (0,0))
            pygame.display.flip()
        
        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
            
        elif mode == "ram":
            return self._create_board_array()
        
    
    def _render_board(self, canvas):
        if self.board_surface is None:
            self.board_surface = pygame.Surface((BOARD_SURFACE_WIDTH, BOARD_SURFACE_HEIGHT))
            
        self.board_surface.fill(0)
         
        #Borders
        points = [(0,0),(BOARD_SURFACE_WIDTH-2,0),(BOARD_SURFACE_WIDTH-2,BOARD_SURFACE_HEIGHT-2),(0,BOARD_SURFACE_HEIGHT-2)]
        pygame.draw.lines(self.board_surface, BORDER_GRAY, True, points, 2);

        for x in range(1, COLUMNS-1):
            for y in range(2, ROWS-1):
                if(self.board[y][x] != 0):
                    tile = pygame.Surface((CELL_SIZE, CELL_SIZE))
                    tile.fill(TETROMINO_COLORS[self.board[y][x]])
                    self.board_surface.blit(tile, ((x-1)*CELL_SIZE, (y-2)*CELL_SIZE))
              
        
        self._render_ghost_piece(self.board_surface)            
        self._render_curr_piece(self.board_surface)
        
        
        canvas.blit(self.board_surface, (PADDING + HOLD_WIDTH, PADDING))
        
    
    def _render_curr_piece(self, board):
        #if self.piece_surface is None:
        piece_array = self._rotate(self.curr_piece_type, self.rotation)
        piece_width, piece_height = piece_array.shape[1], piece_array.shape[0]
        piece_surface = pygame.Surface((piece_width * CELL_SIZE, piece_height * CELL_SIZE), pygame.SRCALPHA)
        
        piece_surface.fill(0)
        
        for x in range(piece_width):
            for y in range(piece_height):
                if(piece_array[y][x] != 0):
                    tile = pygame.Surface((CELL_SIZE, CELL_SIZE))
                    tile.fill(TETROMINO_COLORS[self.curr_piece_type+1])
                    piece_surface.blit(tile, (x * CELL_SIZE, y * CELL_SIZE))
        
        board.blit(piece_surface, ((self.px-1) * CELL_SIZE, (self.py-2) * CELL_SIZE))
        
    
    def _render_ghost_piece(self, board):
        ghost_array = self._rotate(self.curr_piece_type, self.rotation)
        ghost_width, ghost_height = ghost_array.shape[1], ghost_array.shape[0]
        ghost_surface = pygame.Surface((ghost_width * CELL_SIZE, ghost_height * CELL_SIZE), pygame.SRCALPHA)
        ghost_surface.fill(0)
        
        for x in range(ghost_width):
            for y in range(ghost_height):
                if(ghost_array[y][x] != 0):
                    tile = pygame.Surface((CELL_SIZE, CELL_SIZE))
                    color = TETROMINO_COLORS[self.curr_piece_type+1]
                    tile.fill(color)
                    ghost_surface.blit(tile, (x*CELL_SIZE,y*CELL_SIZE))

        ghost_surface.set_alpha(128)
        board.blit(ghost_surface, ((self.ghostX-1) * CELL_SIZE, (self.ghostY-2) * CELL_SIZE))
                
        
            
    def _render_preview(self, canvas):
        if self.preview_surface is None:
            self.preview_surface = pygame.Surface((PREVIEW_WIDTH, PREVIEW_HEIGHT))
        self.preview_surface.fill(0)
        
        for i in range(len(self.queue)):
            next_surf = self._render_next(self.queue[i])
            self.preview_surface.blit(next_surf, (CELL_SIZE, (3*i+1)*CELL_SIZE))
        
        canvas.blit(self.preview_surface, ((PADDING + BOARD_SURFACE_WIDTH + HOLD_WIDTH, PADDING)))
        
    
    def _render_next(self, type):
        piece_array = TETROMINO_default[type]
        width = piece_array.shape[1]
        height = piece_array.shape[0]
        next_surf = pygame.Surface((width*CELL_SIZE, height*CELL_SIZE))
        for px in range(width):
            for py in range(height):
                if(piece_array[py,px] != 0):
                    tile = pygame.Surface((CELL_SIZE, CELL_SIZE))
                    tile.fill(TETROMINO_COLORS[type+1])
                    next_surf.blit(tile, (px * CELL_SIZE, py * CELL_SIZE))
        return next_surf
                    
    
    def _render_stats(self, canvas):
        if self.stat_surface is None:
            self.stat_surface = pygame.Surface((STAT_WIDTH, STAT_HEIGHT))
            self.stat_surface.fill(0)
            timeLabel = FONT.render("Time  ", 5, GRAY)
            scoreLabel = FONT.render("Score  ", 5, GRAY)
            self.stat_surface.blit(timeLabel, (STAT_WIDTH / 2 - timeLabel.get_width(), CELL_SIZE ))
            self.stat_surface.blit(scoreLabel, (STAT_WIDTH / 2 - timeLabel.get_width(), CELL_SIZE * 2 ))
            
        if self.renderTimeCooldown < 100:
            self.renderTimeCooldown += 1000.0 / self.metadata["render_fps"]
        else:
            self.timeLabel = FONT.render(str(format(self.timeElapsed, ".2f")), 5, GRAY)
            self.renderTimeCooldown = 0.0
        self._render_time(self.timeLabel, self.stat_surface)
        
        
        if self.totalScore == 0:
            self.prevTotalScore = 0
            scoreLabel = FONT.render(str(0), 5, GRAY)
        if self.prevTotalScore != self.totalScore:
            scoreLabel = FONT.render(str(self.totalScore), 5, GRAY)
        self._renderScore(scoreLabel, self.stat_surface)
        
        canvas.blit(self.stat_surface, (PADDING + HOLD_WIDTH, PADDING + BOARD_SURFACE_HEIGHT))
        
    
    def _render_time(self, timeLabel, dest):            
        dest.blit(timeLabel, (STAT_WIDTH / 2, CELL_SIZE))
        
                        
    def _renderScore(self, label, dest):
        dest.blit(label, (STAT_WIDTH / 2, CELL_SIZE * 2))

        
    
    def _render_hold(self, canvas):
        if self.hold_surface is None:
            self.hold_surface = pygame.Surface((HOLD_WIDTH, HOLD_HEIGHT))
            self.hold_surface.fill(0)
        
        if self.heldPiece != 7:
            self._render_hold_piece(self.hold_surface)
        
        canvas.blit(self.hold_surface, ((PADDING, PADDING)))
        
    
    def _render_hold_piece(self, dest):
        piece = TETROMINO_default[self.heldPiece]
        for x in range(piece.shape[1]):
            for y in range(piece.shape[0]):
                if piece[y,x] != 0:
                    tile = pygame.Surface((CELL_SIZE, CELL_SIZE))
                    tile.fill(TETROMINO_COLORS[self.heldPiece + 1])
                    dest.blit(tile, ((x+1)*CELL_SIZE, (y+1)*CELL_SIZE))
                    
