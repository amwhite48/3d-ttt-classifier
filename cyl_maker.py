import numpy as np
from visualize import visualize_voxels
import random
import math

# generate a teapot in a 3d np array
w = 16

def generate_tall_cylinder():
    table_array = np.zeros((w,w,w))
    radius = random.uniform(0.25, 0.75)
    height = random.uniform(0.5,1)

    for i in range(w):
        for j in range(w):
            for k in range(w):
                x = map_to_space(i)
                y = map_to_space(j)
                z = map_to_space(k)
                if in_cyl(x,y,z, radius, height):
                    table_array[i,j,k] = 1
    return table_array

# map a number 
def map_to_space(num):
    return (num - 8) / 8

def in_cyl(x, y, z, rad, h):
    dist_from_center = math.sqrt(x*x + y*y)
    return dist_from_center < rad and z < h 


# visualize_voxels(generate_tall_cylinder())
# save 
def generate_cyl_dataset(num_t):
    for i in range(num_t):
        cyl = generate_tall_cylinder()
        filename = "tall_cyl_data/cyl_ex_" + str(i) + ".npy"
        with open(filename, 'wb') as f:
            np.save(filename, cyl)


generate_cyl_dataset(1000)
