import datetime
import matplotlib.patches as patches
from models.geometry_utils import *
from sim.followdogsimulation import *

class FollowDogDynamics:
    @staticmethod
    def forward_dynamics(x, u, timestep):
        """Return updated state in a form of `np.ndnumpy`"""

        " x :  xd, yd, theta,   u : vdx, vdy, w  "

        x_next = np.ndarray(shape=(3,), dtype=float)
        x_next[0] = x[0] + (u[1]*math.cos(x[2]) + u[0]*math.sin(x[2])) * timestep
        x_next[1] = x[1] + (u[1]*math.sin(x[2]) - u[0]*math.cos(x[2])) * timestep
        x_next[2] = x[2]  + u[2] * timestep
        return x_next

    @staticmethod
    def forward_dynamics_opt(timestep):
        """Return updated state in a form of `ca.SX`"""

        x_symbol = ca.SX.sym("x", 3)
        u_symbol = ca.SX.sym("u", 3)

        xd_symbol_next = x_symbol[0] + (u_symbol[1]*ca.cos(x_symbol[2]) + u_symbol[0]*ca.sin(x_symbol[2])) * timestep
        yd_symbol_next = x_symbol[1] + (u_symbol[1]*ca.sin(x_symbol[2]) - u_symbol[0]*ca.cos(x_symbol[2])) * timestep
        theta_symbol_next = x_symbol[2]  + u_symbol[2] * timestep

        state_symbol_next = ca.vertcat(xd_symbol_next, yd_symbol_next, theta_symbol_next)
        return ca.Function("Follow_dog_dynamics", [x_symbol, u_symbol], [state_symbol_next])

    @staticmethod
    def nominal_safe_controller(x, timestep):
        """Return updated state using nominal safe controller in a form of `np.ndnumpy`"""
        u_nom = np.zeros(shape=(3,))
        return FollowDogDynamics.forward_dynamics(x, u_nom, timestep), u_nom

    @staticmethod
    def safe_dist(timestep, vmax, horizon, dist_margin):
        """Return a safe distance outside which to ignore obstacles"""
        # TODO: wrap params
        safe_ratio = 1.0
        return safe_ratio * vmax * timestep * horizon + dist_margin
    

class FollowDogStates:
    def __init__(self, x, u=np.array([0.0, 0.0, 0.0])):
        self._x = x
        self._u = u

    def translation_dog(self):
        return np.array([[self._x[0]], [self._x[1]]])
    
    def rotation(self):
        return np.array(
            [
                [math.cos(self._x[2]), -math.sin(self._x[2])],
                [math.sin(self._x[2]), math.cos(self._x[2])],
            ]
        )


class FollowDogRectangleGeometry:
    def __init__(self, length, width, rear_dist):
        self._length = length
        self._width = width
        self._rear_dist = rear_dist
        self._region = RectangleRegion((-length + rear_dist) / 2, (length + rear_dist) / 2, -width / 2, width / 2)

    def equiv_rep(self):
        return [self._region]

    def get_plot_patch(self, state, alpha=0.5):
        length, width, rear_dist = self._length, self._width, self._rear_dist
        x, y, theta = state[0], state[1], state[2]
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

class FollowDogCircleGeometry:
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



class FollowDogGeometry:
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



class FollowDogSystem(System):
    def get_state(self):
        return self._state._x

    def update(self, unew):
        xnew = self._dynamics.forward_dynamics(self.get_state(), unew, 0.1)
        self._state._x = xnew
        self._time += 0.1

    def logging(self, logger):
        logger._xs.append(self._state._x)
        logger._us.append(self._state._u)
