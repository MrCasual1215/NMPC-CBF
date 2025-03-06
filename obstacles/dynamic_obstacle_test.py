import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/home/sp/Desktop/GuideDog/NMPC-DCBF')))  # 将项目根目录添加到路径

from models.geometry_utils import *
from obstacles.dynamic_obstacle import *

fig, ax = plt.subplots()


origin = [1,2]
length = 2
width = 1
theta = np.pi/2
horizon = 11
door = Door(origin, length, width, theta, horizon, timestep=0.1, period=[0,10])
door.move(timestamp=1, w=1)


regions = door.predict()
for region in regions:
    ax.add_patch(region.get_plot_patch())


plt.xlim([0, 5])
plt.ylim([0, 5])
plt.show()
