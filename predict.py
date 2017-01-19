from __future__ import print_function
import argparse
import numpy as np
import cv2
import random
import math

import chainer
import chainer.links as L

from obstacle import *
from graph import *
from vin import VIN

def get_action(a):
    if a == 0: return -1, -1
    if a == 1: return  0, -1
    if a == 2: return  1, -1
    if a == 3: return -1,  0
    if a == 4: return  1,  0
    if a == 5: return -1,  1
    if a == 6: return  0,  1
    if a == 7: return  1,  1
    return None

def set_state(im):
    mode = 0
    goal = [1, 1]
    pos = [10, 10]

    while mode < 2:
        test_img = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(test_img, tuple(goal), tuple(goal), (0, 0, 1), -1)
        cv2.rectangle(test_img, tuple(pos), tuple(pos), (1, 0, 1), -1)

        cv2.imshow("image", cv2.resize(255 - test_img * 255, (300, 300), interpolation=cv2.INTER_NEAREST))
        key = cv2.waitKey(0)

        if (key == 63234 or key == ord('h')) and mode == 0:
            goal[0] -= 1
        if (key == 63233 or key == ord('j')) and mode == 0:
            goal[1] += 1
        if (key == 63232 or key == ord('k')) and mode == 0:
            goal[1] -= 1
        if (key == 63235 or key == ord('l')) and mode == 0:
            goal[0] += 1
        if (key == 63234 or key == ord('h')) and mode == 1:
            pos[0] -= 1
        if (key == 63233 or key == ord('j')) and mode == 1:
            pos[1] += 1
        if (key == 63232 or key == ord('k')) and mode == 1:
            pos[1] -= 1
        if (key == 63235 or key == ord('l')) and mode == 1:
            pos[0] += 1
        if key == ord('q'):
            return None, None
        if key == 13:
            mode += 1

    return pos, goal

def predict(im, prior, pos, model):
    map_data = np.concatenate(
        (np.expand_dims(im, 0), np.expand_dims(prior, 0)),
        axis=0).astype(dtype=np.float32)
    map = chainer.Variable(np.reshape(map_data, (1,) + map_data.shape))

    s1_data = np.array(pos[0], dtype=np.int32)
    s2_data = np.array(pos[1], dtype=np.int32)
    label_data = np.array([0], dtype=np.int32)

    s1 = chainer.Variable(np.reshape(np.array(s1_data), (1, 1)))
    s2 = chainer.Variable(np.reshape(np.array(s2_data), (1, 1)))
    label = chainer.Variable(np.reshape(np.array(label_data), (1,)))

    model(map, s1, s2, label)

    action = np.argmax(model.predictor.ret.data)
    reward = model.predictor.r.data
    value = model.predictor.v.data
    reward = np.reshape(reward, reward.shape[2:])
    value = np.reshape(value, value.shape[2:])

    return action, reward, value

def main():
    size_1 = 16
    size_2 = 16
    dom_size = (size_1, size_2)
    max_obs = 40
    max_obs_size = 1.0

    parser = argparse.ArgumentParser(description='VIN')
    parser.add_argument('--model', '-m', type=str, default='',
                        help='Model from given file')
    args = parser.parse_args()

    model = L.Classifier(VIN(k=20))
    chainer.serializers.load_npz(args.model, model)

    while True:
        obs = Obstacle(dom_size, (0, 0), max_obs_size)
        n_obs = obs.add_n_obs(random.randint(0, max_obs))
        if n_obs == 0:
            continue
        obs.add_border()

        im = obs.getimage()
        pos, goal = set_state(im)

        if pos is None:
            break

        G = GraphBase(im, tuple(goal))
        prior = G.get_reward_prior()

        action, reward, value = predict(im, prior, pos, model)

        path = [tuple(pos)]
        num_traj = 0
        while prior[pos[1], pos[0]] == 0 and num_traj < 30:
            action, _, _ = predict(im, prior, pos, model)
            dx, dy = get_action(action)
            pos[0] = pos[0] + dx
            pos[1] = pos[1] + dy
            path.append(tuple(pos))
            num_traj = num_traj + 1

        test_img = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        reward = (255 * (reward - np.min(reward)) / (np.max(reward) - np.min(reward))).astype(np.uint8)
        value = (255 * (value - np.min(value)) / (np.max(value) - np.min(value))).astype(np.uint8)
        
        for s in path:
            cv2.rectangle(test_img, (s[0], s[1]), (s[0], s[1]), (1, 0, 0), -1)
        cv2.rectangle(test_img, (path[0][0], path[0][1]), (path[0][0], path[0][1]), (0, 1, 1), -1)
        cv2.rectangle(test_img, (goal[0], goal[1]), (goal[0], goal[1]), (0, 0, 1), -1)
        cv2.imshow("image", cv2.resize(255 - test_img * 255, (300, 300), interpolation=cv2.INTER_NEAREST))
        cv2.imshow("reward", cv2.resize(reward, (300, 300), interpolation=cv2.INTER_NEAREST))
        cv2.imshow("value", cv2.resize(value, (300, 300), interpolation=cv2.INTER_NEAREST))
        cv2.waitKey(0)

if __name__ == "__main__":
    main()
