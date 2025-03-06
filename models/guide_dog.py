import datetime

import matplotlib.patches as patches

from models.geometry_utils import *
from sim.mysimulation import *


offset_x = 0.3    
offset_y = 0.4
alpha = 0.2


class GuideDogDynamics:
    @staticmethod
    def forward_dynamics(x, u, timestep):
        """Return updated state in a form of `np.ndnumpy`"""

        " x : xh, yh, xd, yd, theta,   u : vdx, vdy, w  "

        dis = np.linalg.norm(np.array([x[0],x[1]])- np.array([x[2], x[3]]))
        # print(f"distance between human and dog: {dis:.2f}")

        x_next = np.ndarray(shape=(5,), dtype=float)
        x_next[2] = x[2] + (u[1]*math.cos(x[4]) + u[0]*math.sin(x[4])) * timestep
        x_next[3] = x[3] + (u[1]*math.sin(x[4]) - u[0]*math.cos(x[4])) * timestep
        x_next[4] = x[4]  + u[2] * timestep

        rotation_matrix = np.array([
            [np.cos(x_next[4]), -np.sin(x_next[4])],
            [np.sin(x_next[4]), np.cos(x_next[4])]
        ])
        delta = np.dot(np.array([-offset_y, offset_x]), rotation_matrix.T)

        x_ref = x_next[2] + delta[0]
        y_ref = x_next[3] + delta[1]
        x_next[0] = alpha * x_ref + (1-alpha)*x[0]
        x_next[1] = alpha * y_ref + (1-alpha)*x[1]

        
        return x_next

    @staticmethod
    def forward_dynamics_opt(timestep):
        """Return updated state in a form of `ca.SX`"""



        x_symbol = ca.SX.sym("x", 5)
        u_symbol = ca.SX.sym("u", 3)

        xd_symbol_next = x_symbol[2] + (u_symbol[1]*ca.cos(x_symbol[4]) + u_symbol[0]*ca.sin(x_symbol[4])) * timestep
        yd_symbol_next = x_symbol[3] + (u_symbol[1]*ca.sin(x_symbol[4]) - u_symbol[0]*ca.cos(x_symbol[4])) * timestep
        theta_symbol_next = x_symbol[4]  + u_symbol[2] * timestep

        rotation_matrix = np.array([
            [np.cos(theta_symbol_next), -np.sin(theta_symbol_next)],
            [np.sin(theta_symbol_next), np.cos(theta_symbol_next)]
        ])
        delta = np.dot(np.array([-offset_y, offset_x]), rotation_matrix.T)

        x_ref = xd_symbol_next + delta[0]
        y_ref = yd_symbol_next + delta[1]
        xh_symbol_next = alpha * x_ref + (1-alpha)*x_symbol[0]
        yh_symbol_next = alpha * y_ref + (1-alpha)*x_symbol[1]
        state_symbol_next = ca.vertcat(xh_symbol_next, yh_symbol_next, xd_symbol_next, yd_symbol_next, theta_symbol_next)
        return ca.Function("guide_dog_dynamics", [x_symbol, u_symbol], [state_symbol_next])

    @staticmethod
    def nominal_safe_controller(x, timestep):
        """Return updated state using nominal safe controller in a form of `np.ndnumpy`"""
        u_nom = np.zeros(shape=(3,))
        return GuideDogDynamics.forward_dynamics(x, u_nom, timestep), u_nom

    @staticmethod
    def safe_dist(timestep, vmax, horizon, dist_margin):
        """Return a safe distance outside which to ignore obstacles"""
        # TODO: wrap params
        safe_ratio = 1.0
        return safe_ratio * vmax * timestep * horizon + dist_margin
    
    @staticmethod
    def human2dog(huamn_pos):
        x, y, theta = huamn_pos
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        delta = np.dot(np.array([offset_y, -offset_x]), rotation_matrix.T)
        return np.array([x+delta[0], y+delta[1]])
    
    def dog2human(x, y, theta):
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        delta = np.dot(np.array([-offset_y, offset_x]), rotation_matrix.T)
        return np.array([x+delta[0], y+delta[1]])


class GuideDogStates:
    def __init__(self, x, u=np.array([0.0, 0.0, 0.0])):
        self._x = x
        self._u = u

    def translation_human(self):
        return np.array([[self._x[0]], [self._x[1]]])
    
    def translation_dog(self):
        return np.array([[self._x[2]], [self._x[3]]])
    
    def rotation(self):
        return np.array(
            [
                [math.cos(self._x[4]), -math.sin(self._x[4])],
                [math.sin(self._x[4]), math.cos(self._x[4])],
            ]
        )


class GuideDogRectangleGeometry:
    def __init__(self, length, width, rear_dist):
        self._length = length
        self._width = width
        self._rear_dist = rear_dist
        self._region = RectangleRegion((-length + rear_dist) / 2, (length + rear_dist) / 2, -width / 2, width / 2)

    def equiv_rep(self):
        return [self._region]

    def get_plot_patch(self, state, alpha=0.5):
        length, width, rear_dist = self._length, self._width, self._rear_dist
        x, y, theta = state[2], state[3], state[4]
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
        return patches.Polygon(vertices, closed=True, alpha=alpha, fc="tab:brown", ec="None", linewidth=0.8)




class GuideDogCircleGeometry:
    def __init__(self, r):
        self._r = r
        self._region = CircleRegion(r)

    def equiv_rep(self):
        return [self._region]

    def get_plot_patch(self, state, alpha=0.5):
        x = state[0]
        y = state[1]
        r = self._r
        return patches.Circle((x, y), radius=r, alpha=alpha, fc='None', ec="orange", linewidth=2)


class GuideDogGeometry:
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

    def get_plot_patch(self, state, region_idx, alpha=0.5):
        return self._geometries[region_idx].get_plot_patch(state, alpha)



class GuideDogSystem(System):
    def get_state(self):
        return self._state._x

    def update(self, unew):
        xnew = self._dynamics.forward_dynamics(self.get_state(), unew, 0.1)
        self._state._x = xnew
        self._time += 0.1

    def logging(self, logger):
        logger._xs.append(self._state._x)
        logger._us.append(self._state._u)
