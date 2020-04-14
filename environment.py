from gym.core import Env as Env
from env_config import *
import numpy as np
from collections import namedtuple
import matplotlib._color_data as mcd
import random
import os
import pandas as pd
import logging

if ENV.RENDER:
    import pygame


class PalleteWorld(Env):
    """
    State : (Bin List, Current Snapshot of 2D Pallet (exist, weight) )
    """
    def __init__(self, mode='agent', n_random_fixed=None):

        self.n_random_fixed = n_random_fixed
        self.total_items = []
        self.placed_items = []

        if self.n_random_fixed is not None and self.n_random_fixed > 0:
            for i in range(self.n_random_fixed):
                bins_list = []
                for i, b in enumerate(range(ENV.BIN_MAX_COUNT)):
                    x = random.randint(ENV.BIN_MIN_X_SIZE, ENV.BIN_MAX_X_SIZE)
                    y = random.randint(ENV.BIN_MIN_Y_SIZE, ENV.BIN_MAX_Y_SIZE)
                    w = random.randint(ENV.BIN_MIN_W_SIZE, ENV.BIN_MAX_W_SIZE)
                    bins_list.append((x, y, w))
                self.total_items.append(bins_list)

        if ENV.RENDER:
            os.environ['SDL_VIDEO_WINDOW_POS'] = "%d, %d" % (GUI.WINDOW_POS_X, GUI.WINDOW_POS_Y)
            pygame.init()
            self.total_pixel_row_size = ENV.CELL_SIZE * ENV.ROW_COUNT
            self.total_pixel_col_size = ENV.CELL_SIZE * ENV.COL_COUNT

            # https://www.reddit.com/r/pygame/comments/8sw6r0/pygamecolorname_the_list_of_657_names_you_can_use/
            i = 0
            self.colors = []
            while i < ENV.BIN_MAX_COUNT:
                c = pygame.color.THECOLORS.popitem()[1][0:3]
                if c == GUI.WHITE:
                    pass
                else:
                    self.colors.append(c)
                    i=i+1

        self.reset()

    def seed(self, seed=None):
        # todo : check others
        random.seed(seed)
        np.random.seed(seed)

    def reset(self):
        logging.debug('env reset')

        if ENV.RENDER:
            """
            시각화 관련 정보들 초기화
                """
            self.screen = pygame.display.set_mode([self.total_pixel_row_size, self.total_pixel_col_size])
            self.font = pygame.font.SysFont('consolas', ENV.FONT_SIZE, 1)

        self.bins_list = []
        # fill bin's index on the pallet cells (exist, weight)
        self.p = np.zeros((ENV.ROW_COUNT, ENV.COL_COUNT, 3))

        if self.n_random_fixed is None:
            for i, b in enumerate(range(ENV.BIN_MAX_COUNT)):
                x = random.randint(ENV.BIN_MIN_X_SIZE, ENV.BIN_MAX_X_SIZE)
                y = random.randint(ENV.BIN_MIN_Y_SIZE, ENV.BIN_MAX_Y_SIZE)
                w = random.randint(ENV.BIN_MIN_W_SIZE, ENV.BIN_MAX_W_SIZE)
                self.bins_list.append((x, y, w))
        elif self.n_random_fixed > 0:
            i = random.randint(0, self.n_random_fixed-1)
            self.bins_list = self.total_items[i].copy()
        else:
            assert os.path.isfile(self.data_file)
            # todo : change format of data for environment to use this dataset
            df = pd.read_csv(self.datafile)

        # todo : how to set this value to avoid exhaustive search for validation
        self.start_x = 0
        self.start_y = 0
        self.current_bin_placed = 0
        self.previous_actions = []
        self.placed_items = []
        self.is_bin_placed = False
        self.n_step = 0

        return self.bins_list, self.p

    def is_valid(self, c, r, b: Bin):
        is_exist = False
        is_fall = False
        can_load = True
        is_spacious = True
        is_exist_above = False
        verbose = ENV.VERBOSE

        # todo : for test, it examines every possible cases even if it finds out is not impossible for now.
        if r + b.height > ENV.ROW_COUNT:
            is_spacious = False
            if verbose != 0:
                logging.debug('Not valid. It\'s not spacious over x axis.')
            return False

        if c + b.width > ENV.COL_COUNT:
            is_spacious = False
            if verbose != 0:
                logging.debug('Not valid. It\'s not spacious over y axis.')
            return False

        # todo : to make it real, it adds constraints that any bins can not load if there is any bins above of it.
        if r != ENV.ROW_COUNT -1:
            if not (0. == self.p[0:ENV.ROW_COUNT-r, c:c + b.width, 0]).all():
                if verbose != 0:
                    logging.debug('this is not valid because there are other bins above of it.')
                is_exist_above = True
                return False

        if not (0. == self.p[ENV.ROW_COUNT-r-b.height:ENV.ROW_COUNT-r, c:c + b.width, 0]).all():
            if verbose != 0:
                logging.debug('Not valid. The area is already possessed by other bins.')
            is_exist = True
            return False

        if r >= 1:
            if (b.width-sum((self.p[ENV.ROW_COUNT-r, c:c+b.width, 0]) == 0.)) / b.width >= ENV.LOAD_WIDTH_THRESHOLD:
                # todo : check this constraint working
                is_fail = False
            else:
                if verbose != 0:
                    logging.debug('Not valid. The bin could be fallen lack of support of bottom bins')
                is_fall = True
                return False

        # can_load check
        if c > 0:
            if b.weight > self.p[r:r+b.height, c - 1, 1].any():
                if verbose != 0:
                    logging.debug('Not valid. The bins below can not stand the weight of this bin.')
                can_load = False
                return False

        if is_spacious and not is_exist and not is_fall and can_load and not is_exist_above:
            return True
        else:
            return False

    def step(self, action: Action):
        """
        BIN을 어디에 놓느냐를 환경에서 일부 해준다고 가정해보자
        예를 들면 가능한 위치(실제 x,y)는 환경이 정해주고 에이전트는 이걸
        하단에 먼저 채울려고 하는가 아니면 상단에 쌓을려고 하는가를 고른다

        어떤 아이템이 들어왔는데 더 그 공간안에 못쌓는다면 에피소드는 종료되게 처리

        :param action: bin index, priority x or priority y, rotate
        :return: next state
        """
        self.is_bin_placed = False
        self.n_step += 1

        if action.bin_index not in self.previous_actions:

            b = Bin(*(self.bins_list[action.bin_index]))

            if action.rotate == 1:
                b = Bin(*(b.height,b.width, b.weight))

            # https://stackoverflow.com/questions/2597104/break-the-nested-double-loop-in-python
            class BinPlaced(Exception):pass

            try:
                if action.priority == 0:
                    for y in range(self.start_y, ENV.ROW_COUNT):
                        for x in range(self.start_x, ENV.COL_COUNT):
                            if self.is_valid(x, y, b):
                                self.p[ENV.ROW_COUNT - y - b.height:ENV.ROW_COUNT - y, x:x + b.width, 0] = 1
                                self.p[ENV.ROW_COUNT - y - b.height:ENV.ROW_COUNT - y, x:x + b.width, 1] = b.weight
                                self.p[ENV.ROW_COUNT - y - b.height:ENV.ROW_COUNT - y, x:x + b.width, 2] = action.bin_index
                                raise BinPlaced
                elif action.priority == 1:
                    for x in range(self.start_x, ENV.COL_COUNT):
                        for y in range(self.start_y, ENV.ROW_COUNT):
                            if self.is_valid(x, y, b):
                                self.p[ENV.ROW_COUNT - y - b.height:ENV.ROW_COUNT - y, x:x + b.width, 0] = 1
                                self.p[ENV.ROW_COUNT - y - b.height:ENV.ROW_COUNT - y, x:x + b.width, 1] = b.weight
                                self.p[ENV.ROW_COUNT - y - b.height:ENV.ROW_COUNT - y, x:x + b.width, 2] = action.bin_index
                                raise BinPlaced
                if ENV.VERBOSE != 0:
                    logging.debug('This bin can not be placed.')
            except BinPlaced:
                self.is_bin_placed = True
                self.current_bin_placed += 1
                self.placed_items.append(action.bin_index)

        # TODO : We can add this only when this box is replaced..
        self.previous_actions.append(action.bin_index)

        # mask bins_list
        for i in self.previous_actions:
            self.bins_list[i] = (0,0,0)

        # next_state, reward, done, info
        return (self.bins_list, self.p), self.get_reward(), self.is_done(), {'placed_items':self.placed_items}

    def is_done(self):
        if self.current_bin_placed == ENV.BIN_MAX_COUNT or self.n_step == ENV.EPISODE_MAX_STEP:
            return True
        else:
            return False

    def get_reward(self):
        """
        :return:
        """
        # todo : do more reward engineering for every step
        if REWARD.MODE == 0:
            if self.is_done():
                return self.current_bin_placed / ENV.BIN_MAX_COUNT
            elif not self.is_bin_placed:
                return REWARD.INVALID_ACTION_REWARD
            else:
                return 0
        elif REWARD.MODE == 1:  # sum of placed bins' space
            if self.is_done():
                board = np.asarray(self.p[:, :, 0])
                return board.sum()
            elif not self.is_bin_placed:
                return REWARD.INVALID_ACTION_REWARD
            else:
                return 0
        elif REWARD.MODE == 2:  # number of placed bins
            if self.is_done():
                return len(len((self.p[:, :, 0]) != 0.))
            elif not self.is_bin_placed:
                return REWARD.INVALID_ACTION_REWARD
            else:
                return 0

    def render(self, mode='agent'):
        if not ENV.RENDER:
            return

        # self.screen.fill(GUI.WHITE)

        # # lines drawing
        # for x in range(ENV.COL_COUNT + 1):
        #     pygame.draw.line(self.screen,
        #                      GUI.BLACK,
        #                      (0, x * ENV.CELL_SIZE),
        #                      (self.total_pixel_col_size, x * ENV.CELL_SIZE)
        #                      )
        # for y in range(ENV.ROW_COUNT):
        #     pygame.draw.line(self.screen,
        #                      GUI.BLACK,
        #                      (y * ENV.CELL_SIZE, 0),
        #                      (y * ENV.CELL_SIZE,
        #                       self.total_pixel_row_size)
        #                      )

        # Bin drawing
        # todo : maybe we can assign each different color based on its weight rather than based on index!!
        for y in range(self.start_y, ENV.ROW_COUNT):
            for x in range(self.start_x, ENV.COL_COUNT):

                c_i = int(self.p[y, x, 2])

                pygame.draw.rect(
                    self.screen,
                    GUI.WHITE if c_i == 0 else c_i,
                    [x*ENV.CELL_SIZE, y*ENV.CELL_SIZE, ENV.CELL_SIZE, ENV.CELL_SIZE])

        pygame.display.flip()

    def close(self):
        if ENV.RENDER:
            pygame.quit()

    def get_action(self, state):
        pass


class GameState():
    def __init__(self, board, playerTurn):
        self.board = board
        self.pieces = {'1': 'X', '0': '-', '-1': 'O'}
        self.playerTurn = playerTurn
        self.binary = self._binary()
        self.id = self._convertStateToId()
        self.allowedActions = self._allowedActions()
        self.isEndGame = self._checkForEndGame()
        self.value = self._getValue()
        self.score = self._getScore()

    def _allowedActions(self):
        allowed = []
        for i in range(len(self.board)):
            if i >= len(self.board) - 7:
                if self.board[i] == 0:
                    allowed.append(i)
            else:
                if self.board[i] == 0 and self.board[i + 7] != 0:
                    allowed.append(i)

        return allowed

    def _binary(self):
        currentplayer_position = np.zeros(len(self.board), dtype=np.int)
        currentplayer_position[self.board == self.playerTurn] = 1

        other_position = np.zeros(len(self.board), dtype=np.int)
        other_position[self.board == -self.playerTurn] = 1

        position = np.append(currentplayer_position, other_position)

        return (position)

    def _convertStateToId(self):
        player1_position = np.zeros(len(self.board), dtype=np.int)
        player1_position[self.board == 1] = 1

        other_position = np.zeros(len(self.board), dtype=np.int)
        other_position[self.board == -1] = 1

        position = np.append(player1_position, other_position)

        id = ''.join(map(str, position))

        return id

    def _checkForEndGame(self):
        if np.count_nonzero(self.board) == 42:
            return 1

        for x, y, z, a in self.winners:
            if (self.board[x] + self.board[y] + self.board[z] + self.board[a] == 4 * -self.playerTurn):
                return 1
        return 0

    def _getValue(self):
        # This is the value of the state for the current player
        # i.e. if the previous player played a winning move, you lose
        for x, y, z, a in self.winners:
            if (self.board[x] + self.board[y] + self.board[z] + self.board[a] == 4 * -self.playerTurn):
                return (-1, -1, 1)
        return (0, 0, 0)

    def _getScore(self):
        tmp = self.value
        return (tmp[1], tmp[2])

    def takeAction(self, action):
        newBoard = np.array(self.board)
        newBoard[action] = self.playerTurn

        newState = GameState(newBoard, -self.playerTurn)

        value = 0
        done = 0

        if newState.isEndGame:
            value = newState.value[0]
            done = 1

        return (newState, value, done)

    def render(self, logger):
        for r in range(6):
            logger.info([self.pieces[str(x)] for x in self.board[7 * r: (7 * r + 7)]])
        logger.info('--------------')