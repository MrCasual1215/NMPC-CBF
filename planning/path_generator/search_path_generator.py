import sys
import time
import numpy as np

from planning.path_generator.astar import *


def plot_global_map(path, obstacles):
    fig, ax = plt.subplots()
    for o in obstacles:
        patch = o.get_plot_patch()
        ax.add_patch(patch)
    ax.plot(path[:, 0], path[:, 1])
    ax.set_aspect("equal")
    plt.xlim([-1 * 0.15, 13 * 0.15])
    plt.ylim([0 * 0.15, 7 * 0.15])
    plt.show()
    plt.pause(0.001)


class AstarPathGenerator:
    def __init__(self, grid, quad, margin):
        self._global_path = None
        self._grid = GridMap(bounds=grid[0], cell_size=grid[1], quad=quad)
        self._margin = margin

    def generate_path(self, sys, obstacles, goal_pos):
        graph = GraphSearch(graph=self._grid, obstacles=obstacles, margin=self._margin)
        path = graph.a_star(sys.get_state()[:2], goal_pos)
        self._global_path = np.array([p.pos for p in path])
        print(self._global_path)
        if self._global_path == []:
            print("Global Path not found.")
            sys.exit(1)
        if True:
            plot_global_map(self._global_path, obstacles)
        return self._global_path

    def logging(self, logger):
        logger._paths.append(self._global_path)


class AstarLoSPathGenerator:
    def __init__(self, grid, quad, margin):
        self._global_path = None
        self._grid = GridMap(bounds=grid[0], cell_size=grid[1], quad=quad)
        self._margin = margin
        self._global_path = None


    def generate_path(self, sys, obstacles, start_pos, goal_pos):
        graph = GraphSearch(graph=self._grid, obstacles=obstacles, margin=self._margin)
        t1 = time.time()
        path = graph.a_star(start_pos, goal_pos)
        path = graph.reduce_path(path)
        t2 = time.time()
        global_path = np.array([p.pos for p in path])
        # print("global path:",global_path)
        if global_path == []:
            print("Global Path not found.")
            sys.exit(1)
    
        if self._global_path is None:
            self._global_path = global_path
        else:
            self._global_path = np.concatenate((self._global_path, global_path),axis=0)
        return global_path

    def logging(self, logger):
        logger._paths.append(self._global_path)


class ThetaStarPathGenerator:
    def __init__(self, grid, quad, margin):
        self._global_path = None
        self._grid = GridMap(bounds=grid[0], cell_size=grid[1], quad=False)
        self._margin = margin

    def generate_path(self, sys, obstacles, goal_pos):
        graph = GraphSearch(graph=self._grid, obstacles=obstacles, margin=self._margin)
        path = graph.theta_star(sys.get_state()[:2], goal_pos)
        self._global_path = np.array([p.pos for p in path])
        print(f"global path:",self._global_path)

        if self._global_path == []:
            print("Global Path not found.")
            sys.exit(1)
        if True:
            plot_global_map(self._global_path, obstacles)
        return self._global_path

    def logging(self, logger):
        logger._paths.append(self._global_path)
