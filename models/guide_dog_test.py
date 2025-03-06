import math
import sys
import os
import statistics as st
from matplotlib import animation

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/home/sp/Desktop/GuideDog/NMPC-DCBF')))  # 将项目根目录添加到路径


from control.my_dcbf_optimizer import NmpcDcbfOptimizerParam
from control.my_dcbf_controller import NmpcDcbfController

from models.geometry_utils import *

from models.guide_dog import(
    GuideDogDynamics,
    GuideDogRectangleGeometry,
    GuideDogCircleGeometry,
    GuideDogGeometry,
    GuideDogStates,
    GuideDogSystem
)

from planning.path_generator.search_path_generator import (
    AstarLoSPathGenerator,
    AstarPathGenerator,
    ThetaStarPathGenerator,
)
from planning.trajectory_generator.guide_dog_trajectory_generator import (
    GuideDogTrajectoryGenerator,
)
from sim.mysimulation import Robot, SingleAgentSimulation

def animate_world(simulation):
    fig, ax = plt.subplots(figsize=(20, 12.0))
    ax.set_aspect("equal")
    global_paths = simulation._robot._global_planner_logger._paths
    global_path = global_paths[0]
    ax.plot(global_path[:, 0], global_path[:, 1], "ko--", linewidth=1.5, markersize=4)

    local_paths = simulation._robot._local_planner_logger._trajs
    local_path = local_paths[0]
    (reference_traj_line,) = ax.plot(local_path[:, 0], local_path[:, 1], "-", color="blue", linewidth=3, markersize=4)

    optimized_trajs = simulation._robot._controller_logger._xtrajs
    optimized_traj = optimized_trajs[0]
    (optimized_traj_line,) = ax.plot(
        optimized_traj[:, 0],
        optimized_traj[:, 1],
        "-",
        color="gold",
        linewidth=3,
        markersize=4,
    )

    closedloop_traj = np.vstack(simulation._robot._system_logger._xs)
    for obs in simulation._obstacles:
        obs_patch = obs.get_plot_patch()
        ax.add_patch(obs_patch)

    robot_patch = []
    for i in range(simulation._robot._system._geometry._num_geometry):
        if isinstance(simulation._robot._system._geometry.equiv_rep()[i], ConvexRegion2D):
            robot_patch.append(patches.Polygon(np.zeros((1, 2)), alpha=1.0, closed=True, fc="None", ec="blue", linewidth=2))
            ax.add_patch(robot_patch[i])
        else:
            robot_patch.append(patches.Circle((0, 0), radius=0.3, alpha=1, fc='None', ec="orange", linewidth=2))
            ax.add_patch(robot_patch[i])

    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)
    plt.tight_layout()

    def update(index):
        local_path = local_paths[index]
        reference_traj_line.set_data(local_path[:, 0], local_path[:, 1])
        optimized_traj = optimized_trajs[index]
        optimized_traj_line.set_data(optimized_traj[:, 0], optimized_traj[:, 1])
        # plt.xlabel(str(index))
        for i in range(simulation._robot._system._geometry._num_geometry):
            if isinstance(simulation._robot._system._geometry.equiv_rep()[i], ConvexRegion2D):
                polygon_patch_next = simulation._robot._system._geometry.get_plot_patch(closedloop_traj[index, :], i)
                robot_patch[i].set_xy(polygon_patch_next.get_xy())
            else:
                robot_patch[i].set_center((closedloop_traj[index, :][0], closedloop_traj[index, :][1]))
        if index == len(closedloop_traj) - 1:
            ax.plot(closedloop_traj[:, 0], closedloop_traj[:, 1], "k-", linewidth=3, markersize=4)

    anim = animation.FuncAnimation(fig, update, frames=len(closedloop_traj), interval=1000 * 0.1)
    anim.save("animations/guide_dog.mp4", dpi=300, writer=animation.writers["ffmpeg"](fps=10))


def guide_dog_simulation_test():
    start_s, goal_s, grid, obstacles = create_env()

    # print("start_s",start_s)
    # print("goal_s",goal_s)

    geometry_regions = GuideDogGeometry() # 导盲犬的形状
    geometry_regions.add_geometry(GuideDogRectangleGeometry(length=0.8, width=0.45, rear_dist=0.0)) # 狗的形状
    geometry_regions.add_geometry(GuideDogCircleGeometry(r=0.3)) # 人的形状


    robot = Robot(
        GuideDogSystem(
            state=GuideDogStates(x=start_s),
            geometry=geometry_regions,
            dynamics=GuideDogDynamics(),
        )
    )

    global_path_margin = 0.3
    robot.set_global_planner(AstarLoSPathGenerator(grid, quad=False, margin=global_path_margin))
    robot.set_local_planner(GuideDogTrajectoryGenerator())
    robot.set_controller(NmpcDcbfController(dynamics=GuideDogDynamics(), opt_param=NmpcDcbfOptimizerParam()))

    sim = SingleAgentSimulation(robot, obstacles, np.array([goal_s[0], goal_s[1]]))
    sim.run_navigation(40.0)
    # animate_world(sim)
    print("median: ", st.median(robot._controller._optimizer.solver_times))
    print("std: ", st.stdev(robot._controller._optimizer.solver_times))
    print("min: ", min(robot._controller._optimizer.solver_times))
    print("max: ", max(robot._controller._optimizer.solver_times))
    print("Simulation finished.")


def create_env():
    s = 1.2 # scale of environment
    human_init = [2 * s, 2.7 * s,  0]
    human_goal = [5 * s, 2.9 * s,  0]
    dog_init = GuideDogDynamics.human2dog(human_init)  # x_h, y_h, x_d, y_d, theta
    dog_goal  = GuideDogDynamics.human2dog(human_goal) 
    initial_state = np.array([human_init[0], human_init[1], dog_init[0], dog_init[1], human_init[2]])
    goal_state = np.array([human_goal[0], human_goal[1], dog_goal[0], dog_goal[1], human_goal[2]])
    bounds = ((0.0 * s, 0.0 * s), (14.0 * s, 8.0 * s))
    cell_size = 0.25 * s
    grid = (bounds, cell_size)
    obstacles = []

    # door openning test    
    obstacles.append(RectangleRegion(2.0 * s, 8.0 * s, 1.8 * s, 2.0 * s))
    obstacles.append(RectangleRegion(2.0 * s, 4.5 * s, 3.2 * s, 3.4 * s))
    obstacles.append(RectangleRegion(5.5 * s, 8.0 * s, 3.2 * s, 3.4 * s))


    return initial_state, goal_state, grid, obstacles

if __name__ == "__main__":
    guide_dog_simulation_test()
