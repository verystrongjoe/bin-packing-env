from collections import namedtuple
import torch


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# 문제를 가로/세로/하중을 각 BIN의 속성으로 설정
Bin = namedtuple('Bin', ('width', 'height', 'weight'))

# priority 0 이면 1차원 기준먼저 1이면 2차원 기준 먼저
Action = namedtuple('Action', ('bin_index', 'priority', 'rotate'))

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# GUI
class GUI:
    BLACK = (0, 0, 0)
    # WHITE = (255, 255, 255)
    WHITE = BLACK

    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    GRAY = (128, 128, 128)

    WINDOW_POS_X = 0
    WINDOW_POS_Y = 0


# ENVIRONMENT FOR PALLETE
class ENV:

    RENDER = False
    TICK_INTERVAL = 100  # the smaller it is, the slower the game plays

    # Environment Parameter
    FONT_SIZE = 15
    CAPTION_NAME = '2D Bin Packing simulator by Uk Jo'
    ROW_COUNT = 10
    COL_COUNT = 10
    CELL_SIZE = 50

    # Bins information
    BIN_N_STATE = 3  # pos, weight on 2d

    BIN_MAX_COUNT = 20
    EPISODE_MAX_STEP = 20

    BIN_MIN_X_SIZE = 1
    BIN_MIN_Y_SIZE = 1
    BIN_MAX_X_SIZE = 3
    BIN_MAX_Y_SIZE = 3

    BIN_MIN_W_SIZE = 1
    BIN_MAX_W_SIZE = 1

    # Agent Side
    AGENT_STARTING_POS = [0, 0]
    ACTION_SIZE = 3   # x, y, rotate
    N_EPISODES = 1000000

    # Constraint
    LOAD_WIDTH_THRESHOLD = 0.8  # Ratio

    ACTION_SPACE = (BIN_MAX_COUNT, 2, 2)
    VERBOSE = 0


class REWARD:
    INVALID_ACTION_REWARD = 0
    GOAL_REWARD = 1.0
    MODE = 1

class AGENT:
    DISCOUNT_FACTOR_REWARD = 0.9
    LEARNING_RATE = 0.01
    EPSILON = 0.9
    BATCH_SIZE = 32
    TARGET_UPDATE_INTERVAL = 1000
    GAMMA = 0.9
    REPLAY_MEMORY_SIZE = 10000
    EMBEDDING_DIM = 10
