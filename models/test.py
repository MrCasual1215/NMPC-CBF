import numpy as np
import casadi as ca

x= np.array([
    [1, 2, 3],  # 第0行
    [4, 5, 6],  # 第1行
    [7, 8, 9],  # 第2行
    [10, 11, 12] # 第3行
])


a = np.mean(x,0)
print(a)