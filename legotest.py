import numpy as np

small_x = 5
small_y = 5
big_x = 2*small_x + 1
big_y = 2*small_y + 1
xpos = 4
ypos = 4

start_x = big_x//2 - xpos
start_y = big_y//2 - ypos

arr = np.zeros((big_x, big_y))
for x in range(small_x):
    for y in range(small_y):
        arr[x+start_x,y+start_y] = 1

arr[start_x + xpos, start_y+ypos] = 2



print(arr)