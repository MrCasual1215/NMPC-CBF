
- create_env() # 创建地图
  - 返回 起点、终点、地图栅格信息和障碍物（矩形）

- 创建geometry_regions（车的几何区域）

- 创建 Robot类 包括机器人位置、动力学、几何区域

- 为机器人创建 global planner、local planner 和 controller
    - global planner 初始化：  **这块应该不用修改**
      - 输入：栅格信息、 margin是与障碍物的距离
  
    - local planner 初始化： **待修改**
      可以修改的点：
      - horizon
      - 匀速的速度
    
    - controller  **着重修改**
      - 运动学约束、权重

- 创建simulation类
  - 机器人、障碍物和终点



### controller:

NMPCcontroller:
  - 初始化：KinematicCarDynamics()，NmpcDcbfOptimizerParam()
  NmpcDbcfOptimizer：
    - 初始化，variables={}, costs={}, dynamics_opt=KinematicCarDynamics() # casadi中的动力学约束

    