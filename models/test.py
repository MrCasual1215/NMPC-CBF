import numpy as np
import casadi as ca

x= np.array([
    [1, 2, 3],  # 第0行
    [4, 5, 6],  # 第1行
    [7, 8, 9],  # 第2行
    [10, 11, 12] # 第3行
])


squared_sum = ca.sum1(ca.sqrt((x[0] - x[2])**2 + \
              (x[1] - x[3])**2))

print(squared_sum)