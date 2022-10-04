import numpy as np
from visualize import visualize_voxels
import random
import math

# generate a teapot in a 3d np array
w = 16

def generate_table():
    table_array = np.zeros((w,w,w))
    length = random.uniform(0.5,1)
    width = random.uniform(0.5,1)
    height = random.uniform(0,0.9)
    table_thickness = 0.1

    for i in range(w):
        for j in range(w):
            for k in range(w):
                x = map_to_space(i)
                y = map_to_space(j)
                z = map_to_space(k)
                if in_table_top(x, y, z, length, width, height, table_thickness) or in_legs(x, y, z, length, width, height, table_thickness):
                    table_array[i,j,k] = 1
    return table_array

# map a number 
def map_to_space(num):
    return (num - 8) / 8

def in_table_top(x, y, z, length, width, height, table_thickness):
    return abs(x) < length and abs(y) < width and (height - table_thickness) <= z <= (height + table_thickness)

def in_legs(x, y, z, length, width, height, table_thickness):
    return length - table_thickness < abs(x) < length and width - table_thickness < abs(y) < width and z <= height

# visualize_voxels(generate_table())
# save 
def generate_table_dataset(num_tables):
    for i in range(num_tables):
        table = generate_table()
        filename = "table_data/table_ex_" + str(i) + ".npy"
        with open(filename, 'wb') as f:
            np.save(filename, table)


generate_table_dataset(1000)
