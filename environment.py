from gym.core import Env as Env
from config import *
import numpy as np
from collections import namedtuple
import random
import os
import pandas as pd
import matplotlib._color_data as mcd
import logging
if ENV.RENDER:
    import pygame


class PalleteWorld(Env):
    """
    State : (Bin List, Current Snapshot of 2D Pallet (exist, weight) )
    """
    def __init__(self, mode='agent', data_file=None):

        self.data_file = data_file

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
        self.p = np.ones((ENV.ROW_COUNT, ENV.COL_COUNT, 2)) * -1

        if self.data_file is None:
            for i, b in enumerate(range(ENV.BIN_MAX_COUNT)):
                x = random.randint(ENV.BIN_MIN_X_SIZE, ENV.BIN_MAX_X_SIZE)
                y = random.randint(ENV.BIN_MIN_Y_SIZE, ENV.BIN_MAX_Y_SIZE)
                w = random.randint(ENV.BIN_MIN_W_SIZE, ENV.BIN_MAX_W_SIZE)
                self.bins_list.append((x, y, w))
        else:
            assert os.path.isfile(self.data_file)
            # todo : change format of data for environment to use this dataset
            df = pd.read_csv(self.datafile)

        # todo : how to set this value to avoid exhaustive search for validation
        self.start_x = 0
        self.start_y = 0
        self.current_bin_placed = 0
        self.previous_actions = []
        self.is_bin_placed = False

        return self.bins_list, self.p

    def is_valid(self, c, r, b: Bin):
        is_exist = False
        is_fall = False
        can_load = True
        is_spacious = True
        is_exist_above = False

        # todo : for test, it examines every possible cases even if it finds out is not impossible for now.
        if r + b.height > ENV.ROW_COUNT:
            is_spacious = False
            logging.debug('this is not valid because it is not spacious over x axis.')
            return False

        if c + b.width > ENV.COL_COUNT:
            is_spacious = False
            logging.debug('this is not valid because it is not spacious over y axis.')
            return False

        # todo : to make it real, it adds constraints that any bins can not load if there is any bins above of it.
        if r != ENV.ROW_COUNT -1:
            if not (-1 == self.p[0:ENV.ROW_COUNT-r, c:c + b.width, 0]).all():
                logging.debug('this is not valid because there are other bins above of it.')
                is_exist_above = True

        if not (-1 == self.p[ENV.ROW_COUNT-r-b.height:ENV.ROW_COUNT-r, c:c + b.width, 0]).all():
            logging.debug('this is not valid because this area is already possessed by other bins.')
            is_exist = True

        if r >= 1:
            if (b.width-sum((self.p[ENV.ROW_COUNT-r, c:c+b.width, 0]) == -1.0)) / b.width >= ENV.LOAD_WIDTH_THRESHOLD:
                # todo : check this constraint working
                is_fail = False
            else:
                logging.debug('this is not valid because this bin could be fallen lack of support of bottom bins')
                is_fall = True

        # can_load check
        if c > 0:
            if b.weight > self.p[r:r+b.height, c - 1, 1].any():
                logging.debug('this is not valid because bins below can not stand the weight of this bin.')
                can_load = False

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
                                self.p[ENV.ROW_COUNT - y - b.height:ENV.ROW_COUNT - y, x:x + b.width, 0] = action.bin_index
                                self.p[ENV.ROW_COUNT - y - b.height:ENV.ROW_COUNT - y, x:x + b.width, 1] = b.weight
                                raise BinPlaced
                elif action.priority == 1:
                    for x in range(self.start_x, ENV.COL_COUNT):
                        for y in range(self.start_y, ENV.ROW_COUNT):
                            if self.is_valid(x, y, b):
                                self.p[ENV.ROW_COUNT - y - b.height :ENV.ROW_COUNT - y, x:x + b.width, 0] = action.bin_index
                                self.p[ENV.ROW_COUNT - y - b.height :ENV.ROW_COUNT - y, x:x + b.width, 1] = b.weight
                                raise BinPlaced
                logging.debug('This bin can not be placed.')
            except BinPlaced:
                self.is_bin_placed = True
                self.current_bin_placed += 1
                self.previous_actions.append(action.bin_index)

        # next_state, reward, done, info
        return (self.bins_list, self.p), self.get_reward(), self.is_done(), None

    def is_done(self):
        if self.current_bin_placed == ENV.BIN_MAX_COUNT:
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
                return len(len((self.p[:, :, 0]) != -1.0))
            elif not self.is_bin_placed:
                return REWARD.INVALID_ACTION_REWARD
            else:
                return 0
        elif REWARD.MODE == 2:  # number of placed bins
            if self.is_done():
                return len(len((self.p[:, :, 0]) != -1.0))
            elif not self.is_bin_placed:
                return REWARD.INVALID_ACTION_REWARD
            else:
                return 0

    def render(self, mode='agent'):
        if not ENV.RENDER:
            return
        self.screen.fill(GUI.WHITE)

        # lines drawing
        for x in range(ENV.COL_COUNT + 1):
            pygame.draw.line(self.screen,
                             GUI.BLACK,
                             (0, x * ENV.CELL_SIZE),
                             (self.total_pixel_col_size, x * ENV.CELL_SIZE)
                             )
        for y in range(ENV.ROW_COUNT):
            pygame.draw.line(self.screen,
                             GUI.BLACK,
                             (y * ENV.CELL_SIZE, 0),
                             (y * ENV.CELL_SIZE,
                              self.total_pixel_row_size)
                             )

        # Bin drawing
        # todo : maybe we can assign each different color based on its weight rather than based on index!!
        for y in range(self.start_y, ENV.ROW_COUNT):
            for x in range(self.start_x, ENV.COL_COUNT):
                pygame.draw.rect(
                    self.screen,
                    GUI.WHITE if int(self.p[y][x][0]) == -1 else self.colors[int(self.p[y][x][0])],
                                 [x*ENV.CELL_SIZE,
                                  y*ENV.CELL_SIZE,
                                  ENV.CELL_SIZE,
                                  ENV.CELL_SIZE])
        pygame.display.flip()

    def close(self):
        pygame.quit()

    def get_action(self, state):
        pass

