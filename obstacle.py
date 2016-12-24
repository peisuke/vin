import cv2
import numpy as np
import random
import pickle
import heapq
import math
from collections import defaultdict
from progressbar import ProgressBar

class Obstacle:
    def __init__(self, domsize, goal, max_obs_size):
        self.domsize = domsize
        self.max_obs_size = max_obs_size
        self.goal = goal
        self.im = np.zeros(self.domsize, dtype=np.uint8)
        self.mask = np.zeros(self.domsize, dtype=np.uint8)

    def add_n_obs(self, num_obs):
        res = 0
        for i in xrange(num_obs):
            obs_type = self.set_type()
            ret = self.add_obs(obs_type)
            if ret is True:
                res = res + 1
        return res

    def add_obs(self, obs_type):
        new_im = self.im.copy()

        if obs_type == 'circ':
            rand_rad = int(math.ceil(random.random() * self.max_obs_size))
            randx = int(math.ceil(random.random() * self.domsize[0]))
            randy = int(math.ceil(random.random() * self.domsize[1]))
            cv2.circle(new_im, (randx, randy), rand_rad, 1, -1)
        elif obs_type == 'rect':
            rand_hgt = int(math.ceil(random.random() * self.max_obs_size))
            rand_wid = int(math.ceil(random.random() * self.max_obs_size))
            randx = int(math.ceil(random.random() * self.domsize[0]))
            randy = int(math.ceil(random.random() * self.domsize[1]))
            cv2.rectangle(new_im, (randx, randy), (randx+rand_wid, randy+rand_hgt), 1, -1)
        else:
            print('unexpected obs type.')

        ret = self.check_mask(new_im)
        if ret is True:
            self.im = new_im
            return True
        else:
            return False

    def check_mask(self, new_im):
        if new_im[self.goal[1], self.goal[0]] != 0:
            return False
        else:
            return True

    def set_type(self):
        if random.random() > 0.5:
            return 'rect'
        else:
            return 'circ'

    def add_border(self):
        self.im[:, 0] = 1
        self.im[:, -1] = 1
        self.im[0, :] = 1
        self.im[-1, :] = 1

    def getimage(self):
        return self.im.copy()


