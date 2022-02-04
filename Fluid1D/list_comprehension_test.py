#!/usr/bin/env python3

import numpy as np

def func(x,y,u):
    res = 2*u * (1 - (x**2 + y**2)/5**2)
    return res

x = np.linspace(-5,5,20)
y = np.linspace(-5,5,20)
z = 10

grid = [[x0,y0,z] for y0 in y for x0 in x]

xx,yy = np.meshgrid(x,y)

result = func(xx,yy,1)

result = np.reshape(result, (len(grid),))

size_comparison = len(grid) == xx.size

#print("xx: \n", xx, "\n")
#print("yy: \n", yy, "\n")
#print("grid:", grid, "\n")
#
#print(result)
maximum = np.amax(result)
print(maximum)
print(result[190])
print(np.argmax(result))
#print(np.amin(result))
#print("are they the same size?", size_comparison)

#write_vel = [[0,0,res] if res>=0 else [0,0,0] for res in result]

#print(write_vel)
