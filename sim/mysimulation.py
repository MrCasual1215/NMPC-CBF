import math
import time
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches

from models.geometry_utils import RectangleRegion, ConvexRegion2D
from sim.logger import (
    ControllerLogger,
    GlobalPlannerLogger,
    LocalPlannerLogger,
    SystemLogger,
)


class System:
    def __init__(self, time=0.0, state=None, geometry=None, dynamics=None):
        self._time = time
        self._state = state
        self._geometry = geometry
        self._dynamics = dynamics


class Robot:
    def __init__(self, system):
        self._system = system
        self._system_logger = SystemLogger()

    def set_global_planner(self, global_planner):
        self._global_planner = global_planner
        self._global_planner_logger = GlobalPlannerLogger()

    def set_local_planner(self, local_planner):
        self._local_planner = local_planner
        self._local_planner_logger = LocalPlannerLogger()

    def set_controller(self, controller):
        self._controller = controller
        self._controller_logger = ControllerLogger()

    def run_global_planner(self, sys, obstacles, goal_pos):
        # TODO: global path shall be generated with `system` and `obstacles`.
        self._global_path = self._global_planner.generate_path(sys, obstacles, goal_pos)
        self._global_planner.logging(self._global_planner_logger)

    def run_local_planner(self):
        # TODO: local path shall be generated with `obstacles`.
        self._local_trajectory = self._local_planner.generate_trajectory(self._system, self._global_path)
        self._local_planner.logging(self._local_planner_logger)

    def run_controller(self, obstacles):
        self._control_action = self._controller.generate_control_input(
            self._system, self._global_path, self._local_trajectory, obstacles
        )
        self._controller.logging(self._controller_logger)

    def run_system(self):
        self._system.update(self._control_action)
        self._system.logging(self._system_logger)



class SingleAgentSimulation:
    def __init__(self, robot, obstacles, goal_position):
        self._robot = robot
        self._obstacles = obstacles
        self._goal_position = goal_position
        self.fig, self.ax = plt.subplots(figsize=(20, 12.0))


    def run_navigation(self, navigation_time):
        self._robot.run_global_planner(self._robot._system, self._obstacles, self._goal_position)
        while self._robot._system._time < navigation_time:
            self._robot.run_local_planner()
            self._robot.run_controller(self._obstacles)
            self._robot.run_system()
            self.show()


    def show(self):
        self.ax.cla() 
        self.ax.set_aspect("equal")


        local_path = self._robot._local_planner_logger._trajs[-1]
        optimized_traj = self._robot._controller_logger._xtrajs[-1]
        global_path = self._robot._global_planner_logger._paths[-1]
        closedloop_state = np.vstack(self._robot._system_logger._xs)[-1,:]

        self.ax.plot(global_path[:, 0], global_path[:, 1], "ko--", linewidth=1.5, markersize=4)
        self.ax.plot(local_path[:, 0], local_path[:, 1], "-", color="blue", linewidth=3, markersize=4)
        self.ax.plot(optimized_traj[:, 0], optimized_traj[:, 1], "-", color="gold", linewidth=3, markersize=4,)


        for obs in self._obstacles:
            obs_patch = obs.get_plot_patch()
            self.ax.add_patch(obs_patch)

        robot_patch = []
        for i in range(self._robot._system._geometry._num_geometry):
            if isinstance(self._robot._system._geometry.equiv_rep()[i], ConvexRegion2D):
                robot_patch.append(patches.Polygon(np.zeros((1, 2)), alpha=1.0, closed=True, fc="None", ec="blue", linewidth=2))
                polygon_patch_next = self._robot._system._geometry.get_plot_patch(closedloop_state, i, 0.8)
                robot_patch[i].set_xy(polygon_patch_next.get_xy())
                self.ax.add_patch(robot_patch[i])
            else:
                circle_patch_next = self._robot._system._geometry.get_plot_patch(closedloop_state, i, 0.5)
                self.ax.add_patch(circle_patch_next)
            
        plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)
        plt.tight_layout()
        plt.pause(0.001)




