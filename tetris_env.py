import numpy as np
from settings import *
import gymnasium as gym
from gymnasium import spaces
from collections import deque
from controller import Controller


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
    
    
    Action space: Box(0, 2, shape=(7,) dtype=int)
    [left, right, rotate left, rotate right, soft drop, hard drop, hold]
    Action space: Box(0, 2, shape=(7,) dtype=int)
    [left, right, rotate left, rotate right, soft drop, hard drop, hold]
    
    Observation space: a flattened 1D array of the board.
                
    Different modes available:
        -0: Infinite mode
        -1: 40 lines mode
    
    """
    def __init__(self, game_mode=0, render_mode=None, DAS=DAS, ARR=ARR, train_mode=False):
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
        
        self.gameMode = game_mode
        self.ctrl = Controller(DAS, ARR)
        self.isTraining = train_mode
                
        # self.action_space = spaces.MultiBinary(7)
        self.action_space = spaces.Box(0,1,shape=(7,),dtype=np.float32)

        self.observation_space = spaces.Box(0, 9, shape=(ROWS * COLUMNS,), dtype=int)
    
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.ctrl.reset()
        self.gameOver = False
        self.totalScore = 0
        self.totalSteps = 0
        
        ## Board setup
        self.board = np.zeros((ROWS,COLUMNS), dtype = np.int8)
        
        ## 40 lines mode set up
        self.linesCleared = 0
        self.linesToClear = 40
        self.linesLabel = FONT.render(str(self.linesToClear), 5, GRAY)
        
        
        ## Piece setup
        self.bag = self._7bagRandomizer()
        self._initializePieceQueue()
        self._spawn()
        self._setGhostCoord(self.board, self.curr_piece_type, self.px, self.py, self.rotation)
        self.holdAllowed = True
        self.heldPiece = 7 #none
        
        
        ## Features setup
        self.gravityOn = True
        self.softDropOn = False
        self.startLocking = False
        
        self.previous_dir = 0
        self.lock_t = 0
        self.drop_t = DROP_INTERVAL
        self.soft_t = SOFT_DROP_INTERVAL
        self.force_lock_t = 0
        
        
        ## reward related stuff
        self.numRotations = 0
        self.previousFitness = 0


        ## INFO
        self.timeElapsed = 0.0

        if self.render_mode == 'human':
            self.render()
        
        return self._getObs(), {}
    
        
    def step(self, input):
        """
        input = [left, right, rot left, rot right, soft drop, hard drop, hold]
        """
        x = self.ctrl.update(input)
        obs, reward, terminated, truncated, info = (None, 0, False, False, None)

        if(self.ctrl.justReleasedLeftOrRight([0,0,0],[1,0,0])):
            print("hi")
        self.px += x

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info
    
    
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
    def cont_to_bin(self, contAction):
        return np.round(contAction).astype(int)
    
    
    def _bumpiness(self, board):
        total = 0
        for col in range(1, COLUMNS-2):
            total += abs(self._columnHeight(board, col) - self._columnHeight(board, col+1))
            
        return total
    
    def _aggregateHeight(self, board):
        total = 0
        for col in range(1, COLUMNS-1):
            height = self._columnHeight(board, col)
            total += height
            
        return total
    
        
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
                    
        
    def _getObs(self):
        return self.board.copy().flatten()

    
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
        
    
    def _clearLines(self, board):
        """
        Clears rows that are completed.
        Returns the number of lines cleared.

        Side effect: modifies self.board

        //this chatgpt code is cool
        """
        rows, cols = board.shape
        
        full_rows = np.all(board != 0, axis=1)
        clearCount = np.sum(full_rows)
        if clearCount > 0:
            board = board[~full_rows]
            new_rows = np.zeros((clearCount, COLUMNS), dtype=np.int8)
            board = np.vstack((new_rows, board))
        
        assert board.shape == (ROWS, COLUMNS)

        self.board = board

        return clearCount
    
    
    def _setGhostCoord(self, board, piece, x, y, r):
        """
        Side effect: modifies self.ghostY
        """
        while self._doesFit(board, piece, r, x, y + 1):
            y += 1
        self.ghostY = y
        
        
    def horizontalMove(self, direction, instant=False):
        """
        Side effect: modifies self.px, self.previous_dir
        """
        if instant:
            while self._doesFit(self.board, self.curr_piece_type, self.rotation, self.px+direction, self.py):
                self.px += direction
                self.previous_dir = direction
        else:
            if self._doesFit(self.board, self.curr_piece_type, self.rotation, self.px + direction, self.py):
                self.px += direction
                self.previous_dir = direction
    
    
    def _wallKick(self, board, piece, x, y, r1, r2):
        assert r1 >= 0 and r1 <= 3, 'Illegal rotation value. This should not happen'
        assert r2 >= 0 and r2 <= 3, 'Illegal rotation value. This should not happen'

        if(piece == 0): #I-mino
            translation_vectors = WALL_KICK_DATA_I[(r1, r2)]
            for vector in translation_vectors:
                if(self._doesFit(board, piece, r2, x + vector[0], y - vector[1])):
                    x += vector[0]
                    y -= vector[1]
                    r1 = r2
                    break
        
        elif(piece in [1,2,4,5,6]):
            translation_vectors = WALL_KICK_DATA[(r1, r2)]
            for vector in translation_vectors:
                if(self._doesFit(board, piece, r1, x + vector[0], y - vector[1])):
                    x += vector[0]
                    y -= vector[1]
                    r1 = r2
                    break
        
        self.px = x
        self.py = y
        self.rotation = r1
                

    def _gravity_fall(self, board, piece, x, y, r):
        """
        Side effect: modifies self.py
        """
        shouldLock = True
        if self._doesFit(board, piece, r, x, y):
            y += 1
            shouldLock = False
        
        self.py = y

        return shouldLock
            

    
    def _hard_drop(self):
        """
        Side effect: modifies self.py
        """
        while self._doesFit(self.board, self.curr_piece_type, self.rotation, self.px, self.py + 1):
            self.py += 1
     
    
    def _softDrop(self):
        """
        Side effect: modifies self.py
        """
        shouldLock = True
        if self._doesFit(self.board, self.curr_piece_type, self.rotation, self.px, self.py + 1):
                self.py += 1
                shouldLock = False
            
        return shouldLock

    
    
    def _lock_piece(self, board, piece_type, rotation, x, y):
        """
        Locks the piece on (x,y).

        Side effect: modifies self.board
        """
        piece_array = self._rotate(piece_type, rotation)
        for px in range(piece_array.shape[1]):
            for py in range(piece_array.shape[0]):
                if(piece_array[py,px] != 0):
                    board[y+py][x+px] = piece_type + 1

        

    

    def _swap(self):
        """
        Swaps the current piece with the held piece.
        Basically the same as spawn but does not modify the bag.


        Side effect:
            self.heldPiece
            self.curr_piece_type
            self.queue
            self.px
            self.py
            self.rotation
            self.drop_t
            self.soft_t
            self.lock_t
            self.holdAllowed
            self.gameOver
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
        
        if not self._doesFit(self.board, self.curr_piece_type, self.rotation, self.px, self.py):
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
        self.numRotations = 0
        self.queue.append(next(self.bag))
        
        self.holdAllowed = True
    
        if not self._doesFit(self.board, self.curr_piece_type, self.rotation, self.px, self.py):
            self.gameOver = True

        
        
    def _rotate(self, piece_type, rotation):
        """
        Given a type of piece, returns a 2D array representing the piece after rotation.
        """
        assert rotation >= 0 and rotation <= 3
        return TETROMINO[piece_type][rotation]
    

    def _doesFit(self, board, piece_type, rotation, x, y):
        """
        Returns True if the piece with the given rotation fits into the board at coordinates (x,y).
        The coordinate of the piece is the top-left corner of the piece.
        """
        piece_array = self._rotate(piece_type, rotation)
        r, c = piece_array.shape
        for py in range(r):
            for px in range(c):
                if(piece_array[py,px] != 0):
                    if(x + px < 0 or x + px >= COLUMNS or y + py < 0 or y + py >= ROWS or board[y+py][x+px] != 0):
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
        
        
        self._render_board(self.board, self.canvas)
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
        
    
    def _render_board(self, board, canvas):
        if self.board_surface is None:
            self.board_surface = pygame.Surface((BOARD_SURFACE_WIDTH, BOARD_SURFACE_HEIGHT))
            
        self.board_surface.fill(0)
         
        #Borders
        points = [(0,0),(BOARD_SURFACE_WIDTH-2,0),(BOARD_SURFACE_WIDTH-2,BOARD_SURFACE_HEIGHT-2),(0,BOARD_SURFACE_HEIGHT-2)]
        pygame.draw.lines(self.board_surface, BORDER_GRAY, True, points, 2)

        for x in range(0, COLUMNS):
            for y in range(HIDDEN_ROWS, ROWS):
                if(board[y][x] != 0):
                    tile = pygame.Surface((CELL_SIZE, CELL_SIZE))
                    tile.fill(TETROMINO_COLORS[board[y][x]])
                    self.board_surface.blit(tile, (x*CELL_SIZE, (y-HIDDEN_ROWS)*CELL_SIZE))
              
        
        #self._render_ghost_piece(self.board_surface)            
        self._render_piece(self.curr_piece_type, self.px, self.py, self.rotation, self.board_surface)
        
        canvas.blit(self.board_surface, (PADDING + HOLD_WIDTH, PADDING))
        
    
    def _render_piece(self, piece, x, y, rotation, board_surf):
        #if self.piece_surface is None:
        piece_array = self._rotate(piece, rotation)
        piece_width, piece_height = piece_array.shape[1], piece_array.shape[0]
        piece_surface = pygame.Surface((piece_width * CELL_SIZE, piece_height * CELL_SIZE), pygame.SRCALPHA)
        piece_surface.fill(0)
        ghost_surface = piece_surface.copy()
        ghost_surface.fill(0)
        

        for x in range(piece_width):
            for y in range(piece_height):
                if(piece_array[y][x] != 0):
                    tile = pygame.Surface((CELL_SIZE, CELL_SIZE))
                    tile.fill(TETROMINO_COLORS[self.curr_piece_type+1])
                    piece_surface.blit(tile, (x * CELL_SIZE, y * CELL_SIZE))
                    ghost_surface.blit(tile, (x * CELL_SIZE, y * CELL_SIZE))

        ghost_surface.set_alpha(128)
        
        board_surf.blit(ghost_surface, (self.px * CELL_SIZE, (self.ghostY-HIDDEN_ROWS) * CELL_SIZE))
        board_surf.blit(piece_surface, (self.px * CELL_SIZE, (self.py-HIDDEN_ROWS) * CELL_SIZE))
                
            
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
            linesLabel = FONT.render("Lines Left  ", 5, GRAY)
            self.stat_surface.blit(timeLabel, (STAT_WIDTH / 2 - timeLabel.get_width(), CELL_SIZE ))
            self.stat_surface.blit(scoreLabel, (STAT_WIDTH / 2 - scoreLabel.get_width(), CELL_SIZE * 2 ))
            if(self.gameMode == 1):
                self.stat_surface.blit(linesLabel, (STAT_WIDTH / 2 - linesLabel.get_width(), CELL_SIZE * 3 ))
            
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
        
        
        if(self.gameMode == 1):
            linesToClearLabel = FONT.render(str(self.linesToClear), 5, GRAY)
            self.render_lines_to_clear(linesToClearLabel, self.stat_surface)
        
        canvas.blit(self.stat_surface, (PADDING + HOLD_WIDTH, PADDING + BOARD_SURFACE_HEIGHT))
        
        
    def render_lines_to_clear(self, label, dest):
        dest.blit(label, (STAT_WIDTH / 2, CELL_SIZE * 3))
        
    
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
                    
