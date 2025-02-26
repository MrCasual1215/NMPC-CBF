import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/home/sp/Desktop/GuideDog/NMPC-DCBF')))  # 将项目根目录添加到路径
from planning.trajectory_generator.guide_dog_trajectory_generator import *


class State:
    def __init__(self, x):
        self._x = x


class System:
    def __init__(self, state):
        self._state = state


# # local trajectory generation test
global_path = np.array([[0.0, 0.2], [0.5, 0.2], [0.5, 0.8], [1.0, 0.8]])

# single and repeated waypoints, enpoint test
sys_1 = System(State(np.array([-1.0, -1])))
traj_generator_4 = GuideDogTrajectoryGenerator()
path_4 = traj_generator_4.generate_trajectory(sys_1, global_path)
print(path_4)
