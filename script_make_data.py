import cv2
import numpy as np
import random
import pickle
from progressbar import ProgressBar

from obstacle import *
from graph import *

def sample_graph_traj(graph, num_trajs):
    domsize = graph.domsize

    traj = 0
    states_xy = []
    trial = 0
    while traj < num_trajs:
        trial += 1
        if trial > num_trajs * 2:
            break
        pos = (random.randint(1, domsize[0] - 1), random.randint(1, domsize[1] - 1))
        path = graph.get_shortest_path(pos)
        if path is None:
            continue
        states_xy.append(path)
        traj = traj + 1
        
    return states_xy


def extract_action(traj):
    actions = []
    for i in xrange(len(traj) - 1):
        s0 = traj[i]
        s1 = traj[i+1]
        dx = s1[0] - s0[0]
        dy = s1[1] - s0[1]

        if dx == -1 and dy == -1: actions.append(0)
        if dx == 0 and dy == -1: actions.append(1)
        if dx == 1 and dy == -1: actions.append(2)
        if dx == -1 and dy == 0: actions.append(3)
        if dx == 1 and dy == 0: actions.append(4)
        if dx == -1 and dy == 1: actions.append(5)
        if dx == 0 and dy == 1: actions.append(6)
        if dx == 1 and dy == 1: actions.append(7)

    return actions


def main():
    size_1 = 16
    size_2 = 16
    dom_size = (size_1, size_2)
    max_traj_len = size_1 + size_2
    num_domains = 5000
    max_obs = 40
    max_obs_size = 1.0
    num_trajs = 7
    maxSamples = num_domains * num_trajs * max_traj_len / 2

    im_data = np.zeros((maxSamples, size_2, size_1), dtype=np.uint8)
    value_data = np.zeros((maxSamples, size_2, size_1), dtype=np.uint8)
    state_xy_data = np.zeros((maxSamples, 2), dtype=np.int32)
    label_data = np.zeros((maxSamples), dtype=np.int32)

    prog = ProgressBar(0, num_domains)

    num_samples = 0
    dom = 1
    while dom <= num_domains:
        goal = (random.randint(1, size_1-1), random.randint(1, size_2-1))
        obs = Obstacle(dom_size, goal, max_obs_size)
        n_obs = obs.add_n_obs(random.randint(0, max_obs))
        if n_obs == 0:
            #print('no obstacles added, or problem with border, regenerating map')
            continue
        obs.add_border()

        im = obs.getimage()
        #cv2.imshow("test", cv2.resize(255 - im * 255, (300, 300), interpolation=cv2.INTER_NEAREST))
        #key = cv2.waitKey(0)
        #if key == ord('q'):
        #    break

        G = Graph(im, goal)
        value_prior = G.get_reward_prior()
        states_xy = sample_graph_traj(G, num_trajs)

        if len(states_xy) != num_trajs:
            # print('no trajectory added')
            continue

        for i in xrange(len(states_xy)):
            if len(states_xy[i]) > 0:
                actions = extract_action(states_xy[i])
                ns = len(actions)
                im_data[num_samples:num_samples+ns] = im
                value_data[num_samples:num_samples+ns] = value_prior
                state_xy_data[num_samples:num_samples+ns] = np.array(states_xy[i][0:-1])
                label_data[num_samples:num_samples + ns] = np.array(actions)
                num_samples = num_samples + ns

                #test_img = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
                #cv2.rectangle(test_img, goal, goal, (0, 0, 1), -1)
                #for pos in states_xy[i]:
                #    x = pos[0]
                #    y = pos[1]
                #    cv2.rectangle(test_img, (x,y), (x,y), (1,0,0), -1)
                #    cv2.imshow("test", cv2.resize(255 - test_img * 255, (300, 300), interpolation=cv2.INTER_NEAREST))
                #    key = cv2.waitKey(0)
                #    if key == ord('q'):
                #        return
        prog.update(dom)
        dom = dom + 1

    data = {}
    data['im'] = im_data[0:num_samples]
    data['value'] = value_data[0:num_samples]
    data['state'] = state_xy_data[0:num_samples]
    data['label'] = label_data[0:num_samples]
    with open('map_data.pkl', mode='wb') as f:
        pickle.dump(data,f )

if __name__ == "__main__":
    main()
