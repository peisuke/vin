import numpy as np
import heapq
import math
from collections import defaultdict

class GraphBase(object):
    def __init__(self, image, goal):
        self.domsize = image.shape
        self.nodes = set()
        self.edges = defaultdict(list)
        self.length = {}
        self.set_graph(image)
        self.goal = goal

    def set_graph(self, image):
        # Set node
        for x in xrange(self.domsize[0]):
            for y in xrange(self.domsize[1]):
                if image[y,x] == 0:
                    self.add_node((x,y))

        # Set edge
        for x1 in xrange(self.domsize[0]):
            for y1 in xrange(self.domsize[1]):
                for dx in xrange(-1, 2):
                    for dy in xrange(-1, 2):
                        x2 = x1 + dx
                        y2 = y1 + dy
                        if x2 < 0 or x2 >= self.domsize[0]: continue
                        if y2 < 0 or y2 >= self.domsize[1]: continue
                        if image[y1, x1] == 0 and image[y2, x2] == 0:
                            self.add_halfedge((x1,y1), (x2,y2), math.sqrt(dx*dx+dy*dy))

    def add_node(self, value):
        self.nodes.add(value)

    def add_halfedge(self, from_node, to_node, distance):
        self.edges[from_node].append(to_node)
        self.length[(from_node, to_node)] = distance

    def get_reward_prior(self):
        prior = np.zeros(self.domsize, dtype=np.uint8)
        prior[self.goal[1], self.goal[0]] = 1
        return prior


class Graph(GraphBase):
    def __init__(self, image, goal):
        super(Graph, self).__init__(image, goal)
        self.dist, self.prev = self.dijsktra(goal)

    def dijsktra(self, initial):
        prev = {initial: -1}
        dist = {initial: 0}
        for node in self.nodes:
            if node != initial:
                dist[node] = float('inf')
                prev[node] = -1

        queue = []
        heapq.heappush(queue, (dist[initial], initial))

        while len(queue) > 0:
            current_weight, min_node = heapq.heappop(queue)

            for edge in self.edges[min_node]:
                weight = current_weight + self.length[(min_node, edge)]
                if weight < dist[edge]:
                    dist[edge] = weight
                    prev[edge] = min_node
                    heapq.heappush(queue, (dist[edge], edge))
        return dist, prev

    def get_shortest_path(self, pos):
        path = None
        
        if pos not in self.nodes:
            return None

        if self.prev[pos] != -1:
            path = [pos]
            pos = self.prev[pos]
            path.append(pos)
            while self.prev[pos] != -1:
                pos = self.prev[pos]
                path.append(pos)
            
        return path    

