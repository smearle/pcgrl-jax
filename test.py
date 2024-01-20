import numpy as np

m = np.zeros((3,3))

n = np.zeros((3,3))

xcoord = 2
ycoord = 2

m[0][0] = 1
print(m)

m[0][1] = 1
print(m)

n[xcoord][ycoord]=1
print(n)


def rot90_1(x: int, z: int) -> (int, int):
        return z, 0-x

xmove, ymove = -1, 1
xmove, ymove = rot90_1(xmove, ymove)

print(xmove, ymove)
print(xcoord+xmove, ycoord+ymove)
n[xcoord+xmove][ycoord+ymove] = 1
n = np.rot90(n, k=1, axes=(0,1))
print(n)