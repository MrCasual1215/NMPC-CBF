import datetime
import time

import casadi as ca
import numpy as np

from models.geometry_utils import *


class NmpcDcbfOptimizerParam:
    def __init__(self):
        self.horizon = 11
        self.horizon_dcbf = 6
        self.mat_Q = np.diag([100.0, 100.0, 10, 10, 10])
        self.mat_R = np.diag([0.0, 0.0, 0.0])
        self.mat_Rold = np.diag([1.0, 1.0, 1.0]) * 0.0
        self.mat_dR = np.diag([1.0, 1.0, 1.0]) * 0.0
        self.gamma = 0.8
        self.pomega = 10.0
        self.human_margin_dist = 0.3
        self.dog_margin_dist = 0.3
        self.terminal_weight = 10.0
        self.close2human_weight = 0
        self.max_distance_between_humandog = 1.0
        self.min_distance_between_humandog = 0.5
        self.slack_panishment_weight = 0


class NmpcDbcfOptimizer:
    def __init__(self, variables: dict, costs: dict, dynamics_opt):
        self.opti = None
        self.variables = variables
        self.costs = costs
        self.dynamics_opt = dynamics_opt
        self.solver_times = []

    def set_state(self, state):
        self.state = state

    def initialize_variables(self, param):
        self.variables["x"] = self.opti.variable(5, param.horizon + 1)
        self.variables["u"] = self.opti.variable(3, param.horizon)
        # self.variables["slack_variables"] = self.opti.variable(1)

    def add_initial_condition_constraint(self):
        self.opti.subject_to(self.variables["x"][:, 0] == self.state._x)

    def add_input_constraint(self, param):
        # TODO: wrap params
        vx_max, vy_max, omegamax = 0.3, 0.6, 1.0
        for i in range(param.horizon):
            # input constraints
            self.opti.subject_to(self.variables["u"][0, i] <= vx_max)
            self.opti.subject_to(-vx_max <= self.variables["u"][0, i])
            self.opti.subject_to(self.variables["u"][1, i] <= vy_max)
            self.opti.subject_to(-vy_max <= self.variables["u"][1, i])
            self.opti.subject_to(self.variables["u"][2, i] <= omegamax)
            self.opti.subject_to(-omegamax <= self.variables["u"][2, i])

    def add_input_derivative_constraint(self, param):
        # TODO: Remove this hardcoded function with timestep
        ax_max, ay_max, omegadot_max = 0.8, 1.0, 2
        for i in range(param.horizon - 1):
            # input constraints
            self.opti.subject_to(self.variables["u"][0, i + 1] - self.variables["u"][0, i] <= ax_max)
            self.opti.subject_to(self.variables["u"][0, i + 1] - self.variables["u"][0, i] >= -ax_max)
            self.opti.subject_to(self.variables["u"][1, i + 1] - self.variables["u"][1, i] <= ay_max)
            self.opti.subject_to(self.variables["u"][1, i + 1] - self.variables["u"][1, i] >= -ay_max)
            self.opti.subject_to(self.variables["u"][2, i + 1] - self.variables["u"][2, i] <= omegadot_max)
            self.opti.subject_to(self.variables["u"][2, i + 1] - self.variables["u"][2, i] >= -omegadot_max)
        self.opti.subject_to(self.variables["u"][0, 0] - self.state._u[0] <= ax_max)
        self.opti.subject_to(self.variables["u"][0, 0] - self.state._u[0] >= -ax_max)
        self.opti.subject_to(self.variables["u"][1, 0] - self.state._u[1] <= ay_max)
        self.opti.subject_to(self.variables["u"][1, 0] - self.state._u[1] >= -ay_max)
        self.opti.subject_to(self.variables["u"][2, 0] - self.state._u[2] <= omegadot_max)
        self.opti.subject_to(self.variables["u"][2, 0] - self.state._u[2] >= -omegadot_max)

    def add_dynamics_constraint(self, param):
        for i in range(param.horizon):
            self.opti.subject_to(
                self.variables["x"][:, i + 1] == self.dynamics_opt(self.variables["x"][:, i], self.variables["u"][:, i])
            )

    def add_reference_trajectory_tracking_cost(self, param, reference_trajectory):
        self.costs["reference_trajectory_tracking"] = 0
        for i in range(param.horizon - 1):
            x_diff = self.variables["x"][:, i] - reference_trajectory[i, :]
            self.costs["reference_trajectory_tracking"] += ca.mtimes(x_diff.T, ca.mtimes(param.mat_Q, x_diff))
        x_diff = self.variables["x"][:, -1] - reference_trajectory[-1, :]
        self.costs["reference_trajectory_tracking"] += param.terminal_weight * ca.mtimes(x_diff.T, ca.mtimes(param.mat_Q, x_diff))

    def add_input_stage_cost(self, param):
        self.costs["input_stage"] = 0
        for i in range(param.horizon):
            self.costs["input_stage"] += ca.mtimes(
                self.variables["u"][:, i].T, ca.mtimes(param.mat_R, self.variables["u"][:, i])
            )

    def add_prev_input_cost(self, param):
        self.costs["prev_input"] = 0
        self.costs["prev_input"] += ca.mtimes(
            (self.variables["u"][:, 0] - self.state._u).T,
            ca.mtimes(param.mat_Rold, (self.variables["u"][:, 0] - self.state._u)),
        )

    def add_input_smoothness_cost(self, param):
        self.costs["input_smoothness"] = 0
        for i in range(param.horizon - 1):
            self.costs["input_smoothness"] += ca.mtimes(
                (self.variables["u"][:, i + 1] - self.variables["u"][:, i]).T,
                ca.mtimes(param.mat_dR, (self.variables["u"][:, i + 1] - self.variables["u"][:, i])),
            )

    def add_point_to_convex_constraint(self, param, obs_geo, safe_dist):
        # get current value of cbf
        mat_A, vec_b = obs_geo.get_convex_rep()
        cbf_curr, lamb_curr = get_dist_point_to_region(self.state._x[0:2], mat_A, vec_b)
        # filter obstacle if it's still far away
        if cbf_curr > safe_dist:
            return
        # duality-cbf constraints
        lamb = self.opti.variable(mat_A.shape[0], param.horizon_dcbf)
        omega = self.opti.variable(param.horizon_dcbf, 1)
        for i in range(param.horizon_dcbf):
            self.opti.subject_to(lamb[:, i] >= 0)
            self.opti.subject_to(
                ca.mtimes((ca.mtimes(mat_A, self.variables["x"][0:2, i + 1]) - vec_b).T, lamb[:, i])
                >= omega[i] * param.gamma ** (i + 1) * (cbf_curr - param.human_margin_dist) + param.human_margin_dist
            )
            temp = ca.mtimes(mat_A.T, lamb[:, i])
            self.opti.subject_to(ca.mtimes(temp.T, temp) <= 1)
            self.opti.subject_to(omega[i] >= 0)
            self.costs["decay_rate_relaxing"] += param.pomega * (omega[i] - 1) ** 2
            # warm start
            self.opti.set_initial(lamb[:, i], lamb_curr)
            self.opti.set_initial(omega[i], 0.1)

    def add_convex_to_convex_constraint(self, param, robot_geo, obs_geo, safe_dist):
        mat_A, vec_b = obs_geo.get_convex_rep()
        robot_G, robot_g = robot_geo.get_convex_rep()
        # get current value of cbf

        cbf_curr, lamb_curr, mu_curr = get_dist_region_to_region(
            mat_A,
            vec_b,
            np.dot(robot_G, self.state.rotation().T),
            np.dot(np.dot(robot_G, self.state.rotation().T), self.state.translation_dog()) + robot_g,
        )

        # filter obstacle if it's still far away
        if cbf_curr > safe_dist:
            return
        # duality-cbf constraints
        lamb = self.opti.variable(mat_A.shape[0], param.horizon_dcbf)
        mu = self.opti.variable(robot_G.shape[0], param.horizon_dcbf)
        omega = self.opti.variable(param.horizon, 1)
        for i in range(param.horizon_dcbf):
            robot_R = ca.hcat(
                [
                    ca.vcat(
                        [
                            ca.cos(self.variables["x"][4, i + 1]),
                            ca.sin(self.variables["x"][4, i + 1]),
                        ]
                    ),
                    ca.vcat(
                        [
                            -ca.sin(self.variables["x"][4, i + 1]),
                            ca.cos(self.variables["x"][4, i + 1]),
                        ]
                    ),
                ]
            )
            robot_T = self.variables["x"][2:4, i + 1]
            self.opti.subject_to(lamb[:, i] >= 0)
            self.opti.subject_to(mu[:, i] >= 0)
            self.opti.subject_to(
                -ca.mtimes(robot_g.T, mu[:, i]) + ca.mtimes((ca.mtimes(mat_A, robot_T) - vec_b).T, lamb[:, i])
                >= omega[i] * param.gamma ** (i + 1) * (cbf_curr - 0) + 0
            )
            self.opti.subject_to(
                ca.mtimes(robot_G.T, mu[:, i]) + ca.mtimes(ca.mtimes(robot_R.T, mat_A.T), lamb[:, i]) == 0
            )
            temp = ca.mtimes(mat_A.T, lamb[:, i])
            self.opti.subject_to(ca.mtimes(temp.T, temp) <= 1)
            self.opti.subject_to(omega[i] >= 0)
            self.costs["decay_rate_relaxing"] += param.pomega * (omega[i] - 1) ** 2
            # warm start
            self.opti.set_initial(lamb[:, i], lamb_curr)
            self.opti.set_initial(mu[:, i], mu_curr)
            self.opti.set_initial(omega[i], 0.1)



    def add_dynamic_point_to_convex_constraint(self, param, obs_geos, safe_dist):
        # get current value of cbf
        mat_A, vec_b = obs_geos[0].get_convex_rep()
        cbf_curr, lamb_curr = get_dist_point_to_region(self.state._x[0:2], mat_A, vec_b)
        # filter obstacle if it's still far away
        if cbf_curr > safe_dist:
            return
        # duality-cbf constraints
        lamb = self.opti.variable(mat_A.shape[0], param.horizon_dcbf)
        omega = self.opti.variable(param.horizon_dcbf, 1)
        for i in range(param.horizon_dcbf):
            mat_A, vec_b = obs_geos[i].get_convex_rep()
            self.opti.subject_to(lamb[:, i] >= 0)
            self.opti.subject_to(
                ca.mtimes((ca.mtimes(mat_A, self.variables["x"][0:2, i + 1]) - vec_b).T, lamb[:, i])
                >= omega[i] * param.gamma ** (i + 1) * (cbf_curr - param.human_margin_dist) + param.human_margin_dist
            )
            temp = ca.mtimes(mat_A.T, lamb[:, i])
            self.opti.subject_to(ca.mtimes(temp.T, temp) <= 1)
            self.opti.subject_to(omega[i] >= 0)
            self.costs["decay_rate_relaxing"] += param.pomega * (omega[i] - 1) ** 2
            # warm start
            self.opti.set_initial(lamb[:, i], lamb_curr)
            self.opti.set_initial(omega[i], 0.1)


    def add_dynamic_convex_to_convex_constraint(self, param, robot_geo, obs_geos, safe_dist):
        mat_A, vec_b = obs_geos[0].get_convex_rep()
        robot_G, robot_g = robot_geo.get_convex_rep()
        # get current value of cbf

        cbf_curr, lamb_curr, mu_curr = get_dist_region_to_region(
            mat_A,
            vec_b,
            np.dot(robot_G, self.state.rotation().T),
            np.dot(np.dot(robot_G, self.state.rotation().T), self.state.translation_dog()) + robot_g,
        )

        # filter obstacle if it's still far away
        if cbf_curr > safe_dist:
            return
        print("cbf",param.horizon_dcbf)
        print("len",len(obs_geos))
        # duality-cbf constraints
        lamb = self.opti.variable(mat_A.shape[0], param.horizon_dcbf)
        mu = self.opti.variable(robot_G.shape[0], param.horizon_dcbf)
        omega = self.opti.variable(param.horizon, 1)
        for i in range(param.horizon_dcbf):
            mat_A, vec_b = obs_geos[i].get_convex_rep()
            robot_R = ca.hcat(
                [
                    ca.vcat(
                        [
                            ca.cos(self.variables["x"][4, i + 1]),
                            ca.sin(self.variables["x"][4, i + 1]),
                        ]
                    ),
                    ca.vcat(
                        [
                            -ca.sin(self.variables["x"][4, i + 1]),
                            ca.cos(self.variables["x"][4, i + 1]),
                        ]
                    ),
                ]
            )
            robot_T = self.variables["x"][2:4, i + 1]
            self.opti.subject_to(lamb[:, i] >= 0)
            self.opti.subject_to(mu[:, i] >= 0)
            self.opti.subject_to(
                -ca.mtimes(robot_g.T, mu[:, i]) + ca.mtimes((ca.mtimes(mat_A, robot_T) - vec_b).T, lamb[:, i])
                >= omega[i] * param.gamma ** (i + 1) * (cbf_curr - param.dog_margin_dist) + param.dog_margin_dist
            )
            self.opti.subject_to(
                ca.mtimes(robot_G.T, mu[:, i]) + ca.mtimes(ca.mtimes(robot_R.T, mat_A.T), lamb[:, i]) == 0
            )
            temp = ca.mtimes(mat_A.T, lamb[:, i])
            self.opti.subject_to(ca.mtimes(temp.T, temp) <= 1)
            self.opti.subject_to(omega[i] >= 0)
            self.costs["decay_rate_relaxing"] += param.pomega * (omega[i] - 1) ** 2
            # warm start
            self.opti.set_initial(lamb[:, i], lamb_curr)
            self.opti.set_initial(mu[:, i], mu_curr)
            self.opti.set_initial(omega[i], 0.1)


    def add_obstacle_avoidance_constraint(self, param, system, obstacles_geo, dynamic_obstacles):
        self.costs["decay_rate_relaxing"] = 0
        # TODO: wrap params
        # TODO: move safe dist inside attribute `system`
        safe_dist = system._dynamics.safe_dist(0.1, 1.0, param.horizon_dcbf, param.dog_margin_dist)
        robot_components = system._geometry.equiv_rep()
        for obs_geo in obstacles_geo:
            for robot_comp in robot_components:
                if isinstance(robot_comp, ConvexRegion2D):
                    self.add_convex_to_convex_constraint(param, robot_comp, obs_geo, safe_dist)
                else:
                    self.add_point_to_convex_constraint(param,  obs_geo, safe_dist)

        for dynamic_obstacle in dynamic_obstacles:
            for robot_comp in robot_components:
                if isinstance(robot_comp, ConvexRegion2D):
                    self.add_dynamic_convex_to_convex_constraint(param, robot_comp, dynamic_obstacle.predict(), safe_dist)
                else:
                    pass
                    # self.add_dynamic_point_to_convex_constraint(param,  dynamic_obstacle.predict(), safe_dist)
  



    def add_get_close2human_cost(self, param):
        # 当人自主运动时， 跟随着人
        self.costs["close2human"] = 0
        self.costs["close2human"] +=  param.close2human_weight * (
            ca.mtimes(
                (self.variables["x"][0,:] - self.variables["x"][2,:]),  
                (self.variables["x"][0,:] - self.variables["x"][2,:]).T  
            ) + ca.mtimes(
                (self.variables["x"][1,:] - self.variables["x"][3,:]),  
                (self.variables["x"][1,:] - self.variables["x"][3,:]).T  
            )
        )

        self.costs["slack_panishment"] = 0
        self.costs["slack_panishment"] += param.slack_panishment_weight *  self.variables["slack_variables"]**2

        
        
    def add_get_close2human_constraint(self, param):
        for i in range(param.horizon):
            self.opti.subject_to(
                (self.variables["x"][0,i]-self.variables["x"][2,i])**2 + (self.variables["x"][1,i]-self.variables["x"][3,i])**2 
                  <= (param.max_distance_between_humandog + self.variables["slack_variables"])**2
            )
        ## 加入松弛变量
        self.opti.subject_to(self.variables["slack_variables"] >= 0 ) 


    def add_warm_start(self, param, system):
        # TODO: wrap params
        x_ws, u_ws = system._dynamics.nominal_safe_controller(self.state._x, 0.1)
        for i in range(param.horizon):
            self.opti.set_initial(self.variables["x"][:, i + 1], x_ws)
            self.opti.set_initial(self.variables["u"][:, i], u_ws)

    def setup(self, param, system, reference_trajectory, obstacles, dynamic_obstacles):
        self.set_state(system._state)
        self.opti = ca.Opti()
        self.initialize_variables(param)
        self.add_initial_condition_constraint()
        self.add_input_constraint(param)
        self.add_input_derivative_constraint(param)
        self.add_dynamics_constraint(param)
        self.add_reference_trajectory_tracking_cost(param, reference_trajectory)
        self.add_input_stage_cost(param)
        self.add_prev_input_cost(param)
        self.add_input_smoothness_cost(param)
        self.add_obstacle_avoidance_constraint(param, system, obstacles, dynamic_obstacles)
        # self.add_get_close2human_cost(param)
        # self.add_get_close2human_constraint(param)
        self.add_warm_start(param, system)

    def solve_nlp(self):
        cost = 0
        for cost_name in self.costs:
            cost += self.costs[cost_name]
        self.opti.minimize(cost)
        option = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
        start_timer = datetime.datetime.now()
        self.opti.solver("ipopt", option)
        opt_sol = self.opti.solve()
        end_timer = datetime.datetime.now()
        delta_timer = end_timer - start_timer
        self.solver_times.append(delta_timer.total_seconds())
        print("solver time: ", delta_timer.total_seconds())
        # print("opt slack:",self.opti.value(self.variables["slack_variables"]))
        # print("slack costs:",self.costs["slack_panishment"])

        print()
        return opt_sol
