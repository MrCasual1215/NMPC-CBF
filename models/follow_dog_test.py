import math
import sys
import os
import statistics as st
from matplotlib import animation

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/home/sp/Desktop/GuideDog/NMPC-DCBF')))  # 将项目根目录添加到路径


## -------------- controller--------------##
from control.follow_dcbf_optimizer import NmpcDcbfOptimizerParam
from control.follow_dcbf_controller import NmpcDcbfController

## -------------- model--------------##
from models.geometry_utils import *
from models.follow_dog import(
    FollowDogDynamics,
    FollowDogRectangleGeometry,
    FollowDogStates,
    FollowDogGeometry,
    FollowDogSystem,
    FollowDogCircleGeometry
)

## -------------- global planer--------------##
from planning.path_generator.search_path_generator import (
    AstarLoSPathGenerator
)

## -------------- local planer--------------##
from planning.trajectory_generator.follow_dog_trajectory_generator import (
    FollowDogTrajectoryGenerator
)
## -------------- simulation--------------##
from sim.followdogsimulation import Robot, SingleAgentSimulation

def animate_world(simulation):
    fig, ax = plt.subplots(figsize=(20, 12.0))
    ax.set_aspect("equal")
    global_paths = simulation._robot._global_planner_logger._paths
    global_path = global_paths[0]
    ax.plot(global_path[:, 0], global_path[:, 1], "ko--", linewidth=1.5, markersize=4)

    human_states = simulation._robot._human_trajectory_logger._states
    human_state = human_states[0]
    (human_traj_line,) = ax.plot(human_state[:, 0], human_state[:, 1], "-", color="blue", linewidth=3, markersize=4)


    # local_paths = simulation._robot._local_planner_logger._trajs
    # local_path = local_paths[0]
    # (reference_traj_line,) = ax.plot(local_path[:, 0], local_path[:, 1], "-", color="blue", linewidth=3, markersize=4)



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

    Arrow = patches.FancyArrowPatch((0.0, 0.0), (0.0, 0.0), 
                        arrowstyle='->', mutation_scale=15, color='black')
    
    robot_patch = []
    for i in range(simulation._robot._system._geometry._num_geometry):
        if isinstance(simulation._robot._system._geometry.equiv_rep()[i], ConvexRegion2D):
            robot_patch.append(patches.Polygon(np.zeros((1, 2)), alpha=1.0, closed=True, fc="None", ec="blue", linewidth=2))
            ax.add_patch(robot_patch[i])
            ax.add_patch(Arrow)
        else:
            robot_patch.append(patches.Circle((0, 0), radius=0.3, alpha=1, fc='None', ec="orange", linewidth=2))
            ax.add_patch(robot_patch[i])

    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)
    plt.tight_layout()

    def update(index):
        human_state = human_states[index]
        human_traj_line.set_data(human_state[:, 0], human_state[:, 1])
        optimized_traj = optimized_trajs[index]
        optimized_traj_line.set_data(optimized_traj[:, 0], optimized_traj[:, 1])
        # plt.xlabel(str(index))
        for i in range(simulation._robot._system._geometry._num_geometry):
            if isinstance(simulation._robot._system._geometry.equiv_rep()[i], ConvexRegion2D):
                polygon_patch_next = simulation._robot._system._geometry.get_plot_patch(closedloop_traj[index, :], i)
                robot_patch[i].set_xy(polygon_patch_next.get_xy())
            
                center = np.mean(polygon_patch_next.get_xy(),0)
                arrow = np.mean(polygon_patch_next.get_xy()[0:2],0) 
                Arrow.set_positions((center[0], center[1]), (arrow[0], arrow[1]))  # 更新起点和终点
            else:
                robot_patch[i].set_center((human_states[index][0][0], human_states[index][0][1]))
        if index == len(closedloop_traj) - 1:
            ax.plot(closedloop_traj[:, 0], closedloop_traj[:, 1], "k-", linewidth=3, markersize=4)

    anim = animation.FuncAnimation(fig, update, frames=len(closedloop_traj), interval=1000 * 0.1)
    anim.save("animations/follow_dog.mp4", dpi=300, writer=animation.writers["ffmpeg"](fps=10))


def follow_dog_simulation_test():
    human_s, dog_s, goal_s, grid, obstacles = create_env()
    geometry_regions = FollowDogGeometry() # 导盲犬的形状
    geometry_regions.add_geometry(FollowDogRectangleGeometry(length=0.8, width=0.45, rear_dist=0.0)) # 狗的形状
    geometry_regions.add_geometry(FollowDogCircleGeometry(r=0.3)) # 人的形状

    robot = Robot(
        FollowDogSystem(
            huamn_state=human_s,
            state=FollowDogStates(x=dog_s),
            geometry=geometry_regions,
            dynamics=FollowDogDynamics(),
        )
    )

    global_path_margin = 0.4
    robot.set_global_planner(AstarLoSPathGenerator(grid, quad=False, margin=global_path_margin))
    robot.set_local_planner(FollowDogTrajectoryGenerator())
    robot.set_controller(NmpcDcbfController(dynamics=FollowDogDynamics(), opt_param=NmpcDcbfOptimizerParam()))

    sim = SingleAgentSimulation(robot, obstacles, np.array([goal_s[0], goal_s[1]]))
    sim.run_navigation(35.0)
    animate_world(sim)
    print("median: ", st.median(robot._controller._optimizer.solver_times))
    print("std: ", st.stdev(robot._controller._optimizer.solver_times))
    print("min: ", min(robot._controller._optimizer.solver_times))
    print("max: ", max(robot._controller._optimizer.solver_times))
    print("Simulation finished.")


def create_env():
    s = 1.2 # scale of environment
    goal = [8 * s, 5 * s,  0]
    human_init = np.array([1 * s, 6 * s,  -math.pi/2])
    dog_init = np.array([0.5 * s, 5.5 * s,  math.pi/2])

    bounds = ((0.0 * s, 0.0 * s), (13.0 * s, 6.0 * s))
    cell_size = 0.25 * s
    grid = (bounds, cell_size)
    obstacles = []
    obstacles.append(RectangleRegion(0.0 * s, 3.0 * s, 0.0 * s, 3.0 * s))
    obstacles.append(RectangleRegion(1.0 * s, 2.0 * s, 4.0 * s, 6.0 * s))
    obstacles.append(RectangleRegion(2.0 * s, 6.0 * s, 5.0 * s, 6.0 * s))
    obstacles.append(RectangleRegion(6.0 * s, 7.0 * s, 3.0 * s, 6.0 * s))
    obstacles.append(RectangleRegion(4.0 * s, 5.0 * s, 0.0 * s, 4.0 * s))
    obstacles.append(RectangleRegion(9.0 * s, 10.0 * s, -1.0 * s, 7.0 * s))
    return human_init, dog_init, goal, grid, obstacles

if __name__ == "__main__":
    follow_dog_simulation_test()
