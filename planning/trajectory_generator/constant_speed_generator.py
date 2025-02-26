import numpy as np


class ConstantSpeedTrajectoryGenerator:
    # TODO: Refactor this class to make it light-weight.
    # TODO: Create base class for this local planner
    def __init__(self):
        # TODO: wrap params
        self._global_path_index = 0
        # TODO: number of waypoints shall equal to length of global path
        self._num_waypoint = None
        # local path
        self._reference_speed = 0.2
        self._num_horizon = 11
        self._local_path_timestep = 0.1
        self._local_trajectory = None
        self._proj_dist_buffer = 0.1

    def generate_trajectory(self, system, global_path):
        # TODO: move initialization of _num_waypoint and _global_path to constructor
        if self._num_waypoint is None:
            self._global_path = global_path
            self._num_waypoint = global_path.shape[0]
        pos = system._state._x[0:2]
        # print("pos",pos)
        # TODO: pass _global_path as a reference
        return self.generate_trajectory_internal(pos, self._global_path)

    def generate_trajectory_internal(self, pos, global_path):
        local_index = self._global_path_index
        # print("global_path",global_path)

        # 计算路径长度
        trunc_path = np.vstack([global_path[local_index:, :], global_path[-1, :]])
        print("trunc_path",trunc_path)
        curv_vec = trunc_path[1:, :] - trunc_path[:-1, :]
        # print("curv_vec",curv_vec)
        curv_length = np.linalg.norm(curv_vec, axis=1)
        # print("curv_length",curv_length)

        # 计算速度方向
        if curv_length[0] == 0.0:
            curv_direct = np.zeros((2,))
        else:
            curv_direct = curv_vec[0, :] / curv_length[0]

        
        proj_dist = np.dot(pos - trunc_path[0, :], curv_direct)
        # print("proj_dist",proj_dist)
        # print("curv_length",curv_length)


        if proj_dist >= curv_length[0] - self._proj_dist_buffer and local_index < self._num_waypoint - 1:
            self._global_path_index += 1
            return self.generate_trajectory_internal(pos, global_path)

        # TODO: make the if statement optional
        if proj_dist <= 0.0:
            proj_dist = 0.0

        t_c = (proj_dist + self._proj_dist_buffer) / self._reference_speed
        t_s = t_c + self._local_path_timestep * np.linspace(0, self._num_horizon - 1, self._num_horizon)

        # print("ts",t_s)
        curv_time = np.cumsum(np.hstack([0.0, curv_length / self._reference_speed])) # 到达点时间的累加
        # print("curv_time",curv_time)
        curv_time[-1] += (
            t_c + 2 * self._local_path_timestep * self._num_horizon + self._proj_dist_buffer / self._reference_speed
        )
        # print("curv_time",curv_time)

        path_idx = np.searchsorted(curv_time, t_s, side="right") - 1
        # print("path_idx",path_idx)
        path = np.vstack(
            [
                np.interp(t_s, curv_time, trunc_path[:, 0]),
                np.interp(t_s, curv_time, trunc_path[:, 1]),
            ]
        ).T
        # print("path",path)
        path_vel = self._reference_speed * np.ones((self._num_horizon, 1))
        path_head = np.arctan2(curv_vec[path_idx, 1], curv_vec[path_idx, 0]).reshape(self._num_horizon, 1)
        self._local_trajectory = np.hstack([path, path_vel, path_head])
        # print("self._local_trajectory",self._local_trajectory)
        return self._local_trajectory

    def logging(self, logger):
        logger._trajs.append(self._local_trajectory)
