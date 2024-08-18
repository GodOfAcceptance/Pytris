import tetris_env
from settings import *
import numpy as np
import itertools

class InvalidActionException(Exception):
    pass

class InvalidStateException(Exception):
    pass

env = tetris_env.TetrisEnv( render_mode="human")
class RandomAgent:
    def __init__(self):
        pass
    
    def predict(self, obs):
        return [env.action_space.sample()]
    
class KeepDropAgent:
    def __init__(self):
        pass
    
    def predict(self, obs):
        return [[0,0,0,0,1,0,0]]
            


class PathFinder:
    """
    A path finder that finds the shortest sequence of actions to reach a certain state.
    
    Action: spaces.MultiDiscrete([3,3,3]) #Horizontal, Rotation, Drop Type
    State: 5-tuple: (board, piece, rotation, x, y)
    
    """
    def __init__(self):
        #Omit hold for now.
        self.actionSpace = list(itertools.product(
            [0, 1, 2],  # horizontal
            [0, 1, 2],  # rotation
            [0, 1, 2]   # drop
        ))
            
    
    def getNeighbors(self, state):
        if not self.isValidState(state):
            raise InvalidStateException("Invalid state")
        
        neighbors = []
        for action in self.actionSpace:
            neighbor = self.applyAction(state, action)
            if self.isValidState(neighbor):
                neighbors.append(neighbor)
                
        return neighbors
    
    
    def applyAction(self, state, action):
        board, piece, r, x, y = state
        horizontalDirection, rotationDirection, drop = self.convertInput(action)
        
        new_x = x
        new_y = y
        new_r = r
        
        if drop == 2:
            while self.doesFit((board, piece, r, x, y+1)):
                y += 1
                
            new_board = self.lockPiece(board, piece, r, x, y)
            return (new_board, piece, r, x, y)

    
    def lockPiece(self, board, piece, rotation, x, y):
        """
        Returns a new board with the piece placed at the given coordinates.
        """
        new_board = board.copy()
        piece_array = self.rotate(piece, rotation)
        
        for px in range(piece_array.shape[1]):
            for py in range(piece_array.shape[0]):
                if(piece_array[py,px] != 0):
                    new_board[y+py][x+px] = piece + 1
                            
        if self.isValidState((new_board, piece, rotation, x, y)):
            return new_board
        else:
            raise InvalidStateException("Invalid state after setting piece")
        
    
    def isValidState(self, state):
        board, piece, rotation, x, y = state
        if board.shape != (ROWS, COLUMNS):
            print("Invalid board shape: ", board.shape)
            return False
        if not np.all(board[ROWS-1] == 9):
            print("Invalid board bottom row: ", board[ROWS-1])
            return False
        if not np.all(board[:,0] == 9):
            print("Invalid board left column: ", board[:,0])
            return False
        if not np.all(board[:,COLUMNS-1] == 9):
            print("Invalid board right column: ", board[:,COLUMNS-1])
            return False
        if x < 0 or x >= COLUMNS:
            print("Piece out of bounds x: ", x)
            return False
        if y < 0 or y >= ROWS:
            print("Piece out of bounds y: ", x)
            return False
        if rotation < 0 or rotation >= 4:
            print("Invalid rotation: ", rotation)
            return False
        if piece < 0 or piece >= 7:
            print("Invalid piece: ", piece)
            return False
        
        return True

    def convertInput(self, rawAction):
        """
        Maps the action space into gamestate's interpretable values.
        In particular, the domain [0,1,2] is mapped to [-1,0,1] for horizontal movement and rotations.
        Raises InvalidActoinException if the action is invalid.
        """
        if len(rawAction) != 3:
            raise InvalidActionException("Invalid action length")
        
        trueAction = [0,0,0]
        for i in range(3):
            if rawAction[i] not in [0,1,2]:
                raise InvalidActionException("Invalid action value")
            else:
                if i < 2:
                    trueAction[i] = 0 if rawAction[i] == 0 else -1 if rawAction[i] == 1 else 1
                else:
                    trueAction[i] = rawAction[i]
                    
        return trueAction
    
    
    def doesFit(self, board, piece, rotation, x, y):
        """
        Returns True if the piece with the given rotation fits into the board at coordinates (x,y).
        The coordinate of the piece is the top-left corner of the piece.
        """
        rows, cols = piece.shape
        piece_array = self.rotate(piece, rotation)
        for px in range(piece_array.shape[1]):
            for py in range(piece_array.shape[0]):
                if(px + x >= 0 and px + x < cols):
                    if(py + y >= 0 and py + y < rows):
                        if(piece_array[py,px] != 0 and board[y+py][x+px] != 0):
                            return False
                        
        return True
    
    

    def rotate(self, piece, rotation):
        """
        Given a type of piece, returns a 2D array representing the piece after rotation.
        """
        assert rotation >= 0 and rotation <= 3
        # return np.rot90(TETROMINO[piece_type], k=rotation)
        return TETROMINO[piece][rotation]