import math
import sys
import statistics as st
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/home/sp/Desktop/GuideDog/NMPC-DCBF')))  # 将项目根目录添加到路径


from control.dcbf_optimizer import NmpcDcbfOptimizerParam
from control.dcbf_controller import NmpcDcbfController

from models.geometry_utils import *
from models.kinematic_car import (
    KinematicCarDynamics,
    KinematicCarRectangleGeometry,
    KinematicCarMultipleGeometry,
    KinematicCarStates,
    KinematicCarSystem,
)
from planning.path_generator.search_path_generator import (
    AstarLoSPathGenerator,
    AstarPathGenerator,
    ThetaStarPathGenerator,
)
from planning.trajectory_generator.constant_speed_generator import (
    ConstantSpeedTrajectoryGenerator,
)
from sim.simulation import Robot, SingleAgentSimulation



def kinematic_car_rectangle_simulation_test(maze_type, robot_shape):
    start_pos, goal_pos, grid, obstacles = create_env(maze_type)

    geometry_regions = KinematicCarMultipleGeometry() # 车的轮廓
    geometry_regions.add_geometry(KinematicCarRectangleGeometry(0.8, 0.45, 0.0))


    robot = Robot(
        KinematicCarSystem(
            state=KinematicCarStates(x=np.block([start_pos[:2], np.array([0.0, start_pos[2]])])),
            geometry=geometry_regions,
            dynamics=KinematicCarDynamics(),
        )
    )

    global_path_margin = 0.2
    robot.set_global_planner(AstarLoSPathGenerator(grid, quad=False, margin=global_path_margin))
    robot.set_local_planner(ConstantSpeedTrajectoryGenerator())
    robot.set_controller(NmpcDcbfController(dynamics=KinematicCarDynamics(), opt_param=NmpcDcbfOptimizerParam()))

    sim = SingleAgentSimulation(robot, obstacles, goal_pos)
    sim.run_navigation(20.0)
    print("median: ", st.median(robot._controller._optimizer.solver_times))
    print("std: ", st.stdev(robot._controller._optimizer.solver_times))
    print("min: ", min(robot._controller._optimizer.solver_times))
    print("max: ", max(robot._controller._optimizer.solver_times))
    print("Simulation finished.")


def create_env(env_type):
    s = 0.8  # scale of environment
    start = np.array([0.5 * s, 5.5 * s, -math.pi / 2.0])
    goal = np.array([12.5 * s, 0.5 * s])
    bounds = ((0.0 * s, 0.0 * s), (13.0 * s, 6.0 * s))
    cell_size = 0.25 * s
    grid = (bounds, cell_size)
    obstacles = []
    obstacles.append(RectangleRegion(0.0 * s, 3.0 * s, 0.0 * s, 3.0 * s))
    obstacles.append(RectangleRegion(1.0 * s, 2.0 * s, 4.0 * s, 6.0 * s))
    obstacles.append(RectangleRegion(2.0 * s, 6.0 * s, 5.0 * s, 6.0 * s))
    obstacles.append(RectangleRegion(6.0 * s, 7.0 * s, 4.0 * s, 6.0 * s))
    obstacles.append(RectangleRegion(4.0 * s, 5.0 * s, 0.0 * s, 4.0 * s))
    obstacles.append(RectangleRegion(13.0 * s, 14.0 * s, -1.0 * s, 7.0 * s))
    return start, goal, grid, obstacles

if __name__ == "__main__":
    kinematic_car_rectangle_simulation_test("maze", "rectangle")
