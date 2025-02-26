import datetime

import matplotlib.patches as patches

from models.geometry_utils import *
from sim.simulation import *


# 车的动力学约束、形状

class KinematicCarDynamics:
    @staticmethod
    def forward_dynamics(x, u, timestep):
        """Return updated state in a form of `np.ndnumpy`"""
        l = 0.1
        x_next = np.ndarray(shape=(4,), dtype=float)
        x_next[0] = x[0] + x[2] * math.cos(x[3]) * timestep
        x_next[1] = x[1] + x[2] * math.sin(x[3]) * timestep
        x_next[2] = x[2] + u[0] * timestep
        x_next[3] = x[3] + x[2] * math.tan(u[1]) / l * timestep
        return x_next

    @staticmethod
    def forward_dynamics_opt(timestep):
        """Return updated state in a form of `ca.SX`"""
        l = 0.1
        x_symbol = ca.SX.sym("x", 4)
        u_symbol = ca.SX.sym("u", 2)
        x_symbol_next = x_symbol[0] + x_symbol[2] * ca.cos(x_symbol[3]) * timestep
        y_symbol_next = x_symbol[1] + x_symbol[2] * ca.sin(x_symbol[3]) * timestep
        v_symbol_next = x_symbol[2] + u_symbol[0] * timestep
        theta_symbol_next = x_symbol[3] + x_symbol[2] * ca.tan(u_symbol[1]) / l * timestep
        state_symbol_next = ca.vertcat(x_symbol_next, y_symbol_next, v_symbol_next, theta_symbol_next)
        return ca.Function("dubin_car_dynamics", [x_symbol, u_symbol], [state_symbol_next])

    @staticmethod
    def nominal_safe_controller(x, timestep, amin, amax):
        """Return updated state using nominal safe controller in a form of `np.ndnumpy`"""
        u_nom = np.zeros(shape=(2,))
        u_nom[0] = np.clip(-x[2] / timestep, amin, amax)
        return KinematicCarDynamics.forward_dynamics(x, u_nom, timestep), u_nom

    @staticmethod
    def safe_dist(x, timestep, amin, amax, dist_margin):
        """Return a safe distance outside which to ignore obstacles"""
        # TODO: wrap params
        safe_ratio = 1.25
        brake_min_dist = (abs(x[2]) + amax * timestep) ** 2 / (2 * amax) + dist_margin
        return safe_ratio * brake_min_dist + abs(x[2]) * timestep + 0.5 * amax * timestep ** 2


class KinematicCarStates:
    def __init__(self, x, u=np.array([0.0, 0.0])):
        self._x = x
        self._u = u

    def translation(self):
        return np.array([[self._x[0]], [self._x[1]]])

    def rotation(self):
        return np.array(
            [
                [math.cos(self._x[3]), -math.sin(self._x[3])],
                [math.sin(self._x[3]), math.cos(self._x[3])],
            ]
        )


class KinematicCarRectangleGeometry:
    def __init__(self, length, width, rear_dist):
        self._length = length
        self._width = width
        self._rear_dist = rear_dist
        self._region = RectangleRegion((-length + rear_dist) / 2, (length + rear_dist) / 2, -width / 2, width / 2)

    def equiv_rep(self):
        return [self._region]

    def get_plot_patch(self, state, i, alpha):
        length, width, rear_dist = self._length, self._width, self._rear_dist
        x, y, theta = state[0], state[1], state[3]
        xc = x + (rear_dist / 2) * math.cos(theta)
        yc = y + (rear_dist / 2) * math.sin(theta)
        vertices = np.array(
            [
                [
                    xc + length / 2 * np.cos(theta) - width / 2 * np.sin(theta),
                    yc + length / 2 * np.sin(theta) + width / 2 * np.cos(theta),
                ],
                [
                    xc + length / 2 * np.cos(theta) + width / 2 * np.sin(theta),
                    yc + length / 2 * np.sin(theta) - width / 2 * np.cos(theta),
                ],
                [
                    xc - length / 2 * np.cos(theta) + width / 2 * np.sin(theta),
                    yc - length / 2 * np.sin(theta) - width / 2 * np.cos(theta),
                ],
                [
                    xc - length / 2 * np.cos(theta) - width / 2 * np.sin(theta),
                    yc - length / 2 * np.sin(theta) + width / 2 * np.cos(theta),
                ],
            ]
        )
        return patches.Polygon(vertices, alpha=alpha, closed=True, fc="blue", ec="None", linewidth=0.5)


class KinematicCarMultipleGeometry:
    def __init__(self):
        self._num_geometry = 0
        self._geometries = []
        self._regions = []

    def equiv_rep(self):
        return self._regions

    def add_geometry(self, geometry):
        self._geometries.append(geometry)
        self._regions.append(geometry._region)
        self._num_geometry += 1

    def get_plot_patch(self, state, region_idx, alpha):
        return self._geometries[region_idx].get_plot_patch(state, region_idx, alpha)



class KinematicCarSystem(System):
    def get_state(self):
        return self._state._x

    def update(self, unew):
        xnew = self._dynamics.forward_dynamics(self.get_state(), unew, 0.1)
        self._state._x = xnew
        self._time += 0.1

    def logging(self, logger):
        logger._xs.append(self._state._x)
        logger._us.append(self._state._u)
