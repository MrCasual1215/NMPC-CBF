from models.geometry_utils import *
from abc import ABC, abstractmethod

class DynamicObstacleControl(ABC):
    def __init__(self, horizon, timestep, period) -> None:
        self._horizon = horizon
        self._timestep = timestep
        self._period = period

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def move(self):
        pass

    def get_plot_patch(self):
        return self._region.get_plot_patch()




class Door(DynamicObstacleControl):
    def __init__(self, origin, length, width, theta, horizon, timestep, period, ) -> None:
        super().__init__(horizon, timestep, period)
        self._theta = theta
        self._width = width
        self._length = length
        self._origin =  origin
        self._region = PolytopeRegion.convex_hull(self.forward_dynamics(self._theta))


    def predict(self):
        predicted_patches = []
        for i in range(self._horizon):
            predicted_pose = self.forward_dynamics(i*self._timestep*self.w + self._theta)
            predicted_patch = PolytopeRegion.convex_hull(predicted_pose)
            predicted_patches.append(predicted_patch)
        return predicted_patches
        
    def move(self, timestamp, w):
        self.w = 0
        if timestamp > self._period[0] and timestamp < self._period[1]:
            self.w = w
            self._theta = self._theta + w*self._timestep
        next_pose = self.forward_dynamics(self._theta)
        self._region = PolytopeRegion.convex_hull(next_pose)
        return next_pose


    def forward_dynamics(self, theta):
        x = np.array([
            [self._length/2, -self._width/2],
            [self._length/2, self._width/2],
            [-self._length/2, self._width/2],
            [-self._length/2, -self._width/2]
            ])
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        x = x@rotation_matrix.T
        x += np.array([self._length/2, 0])@rotation_matrix.T + self._origin
        return x
