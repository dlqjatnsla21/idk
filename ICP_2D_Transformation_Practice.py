import matplotlib.pyplot as plt, numpy as np
import math as m
from sympy import Symbol, solve

P1 = np.array([[0,1,1],[-1,0,1],[8,0,1]])
# P2 = np.array([[-1,0,1],[0,-1,1],[0,8,1]])

theta, dx = m.pi/180*45,5
R = np.array([[m.cos(theta),-m.sin(theta),0],[m.sin(theta),m.cos(theta),0],[0,0,1]])
T = np.array([[1,0,dx],[0,1,0],[0,0,1]])
P2 = np.dot(T,np.dot(R,P1.T)).T

P2_x_total = np.sum(P2[:,0])
P2_y_total = np.sum(P2[:,1])
P1_x_total = np.sum(P1[:,0])
P1_y_total = np.sum(P1[:,1])

c = Symbol('c')
s = Symbol('s')
x = Symbol('x')
equation1 = P1_x_total*c - P1_y_total*s + P1.shape[0]*x - P2_x_total
equation2 = P1_x_total*s + P1_y_total*c - P2_y_total
equation3 = c*c+s*s-1
result = solve((equation1,equation2,equation3),dict=True)
print(result)

cos_1 = result[0][c]
sin_1 = result[0][s]
x_1 = result[0][x]
T_1 = np.array([[1,0,x_1],[0,1,0],[0,0,1]])
R_1 = np.array([[cos_1,-sin_1,0],[sin_1,cos_1,0],[0,0,1]])
Test_P2_1 = np.dot(T_1,np.dot(R_1,P1.T)).T

cos_2 = result[1][c]
sin_2 = result[1][s]
x_2 = result[1][x]
T_2 = np.array([[1,0,x_2],[0,1,0],[0,0,1]])
R_2 = np.array([[cos_2,-sin_2,0],[sin_2,cos_2,0],[0,0,1]])
Test_P2_2 = np.dot(T_2,np.dot(R_2,P1.T)).T

if np.max(Test_P2_1-P2) > np.max(Test_P2_2-P2):
    cos = result[1][c]
    sin = result[1][s]
    x = result[1][x]

fig, ax = plt.subplots()
# ax.set_xlim(-10,15)
# ax.set_ylim(-10,15)
ax.scatter(P1[:,0], P1[:,1],c='r', s=10, alpha=1)
ax.scatter(P2[:,0], P2[:,1],c='b', s=10, alpha=1)
ax.scatter(Test_P2_1[:,0], Test_P2_1[:,1],c='g', s=10, alpha=1)
ax.grid(True)
plt.show()