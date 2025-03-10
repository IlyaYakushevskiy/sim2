"""Instructions
Implement a particle class and a cell class
Implement the partitioning of particles function we introduced in class. 
The hard part is making sure your partition function is really bomb proof, check all “edge cases” (e.g., no particles in the cell, all particles on one side or other of the partition, already partitioned data, particles in the inverted order of the partition, etc…). Write boolean test functions for each of these cases.
Call all test functions in sequence and check if they all succeed.
Once you have this, then recursively partition the partitions and build cells linked into a tree as you go. Partition alternately in x and y dimensions, or simply partition the longest dimension of the given cell.
Create a random distribution of particles in 2D and build a tree from the particles.
Plot the particles and tree cells.
"""
import numpy as np
import numpy.random as rng
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle




class Particle: 
    def __init__(self,coord):
        self.coord = coord
    def __str__(self):
        return f"Particle:\n\tLocation: {self.coord}"

class Cell: 
    def __init__(self, c_low, c_high, i_lower, i_upper): 
        self.c_low = c_low #l left corner
        self.c_high = c_high #r up corner
        self.l = None # left child 
        self.r = None # right child
        self.i_lower = i_lower # lower indices position (integer)
        self.i_upper = i_upper # upper indices position (integer)
        
    def __str__(self):
        return f"Cell:\n\tLocation: ({self.c_low}, {self.c_high})\n\tIndex: [{self.i_lower}:{self.i_upper}]"    
    def print_all_cells(self):
        print(self)
        if self.l != None:
            self.l.print_all_cells()
        if self.r != None:
            self.r.print_all_cells()

particles = np.array([Particle([rng.random(), rng.random()]) for i in range(1000)])
cell = Cell([0,0], [1,1], 0, len(particles))

cell.print_all_cells()

def partitioning(particles, cell: Cell, cut_dim: int):


    if cell.i_lower >= cell.i_upper:
        print("Warning: Tried to partition an empty cell. This should not happen.")
        return particles, cell, 0, 0

    
    sorted_indices = sorted(range(cell.i_lower, cell.i_upper), key=lambda i: particles[i].coord[cut_dim])
    median_index = sorted_indices[len(sorted_indices) // 2]
    c_mid = particles[median_index].coord[cut_dim]


    l, h = cell.i_lower, cell.i_upper - 1

    while l <= h:
        if particles[l].coord[cut_dim] > c_mid:
            particles[l], particles[h] = particles[h], particles[l]
            h -= 1  
        else:
            l += 1  

    i_mid = l  

    
    n_l = i_mid - cell.i_lower
    n_r = cell.i_upper - i_mid

    
    l_low, l_high = cell.c_low[:], cell.c_high[:]  
    r_low, r_high = cell.c_low[:], cell.c_high[:]

    l_high[cut_dim] = c_mid  
    r_low[cut_dim] = c_mid   

    
    cell.l = Cell(l_low, l_high, cell.i_lower, i_mid)
    cell.r = Cell(r_low, r_high, i_mid, cell.i_upper)

    return particles, cell, n_l, n_r



def recursive_partitioning(particles, cell, cut_dim, min_num=8):

    particles, cell, l_num, r_num = partitioning(particles, cell, cut_dim)
    
    if l_num > min_num:
        recursive_partitioning(particles, cell.l, (cut_dim+1)%2) #alternates 0/1 each call 

    if r_num > min_num:
        recursive_partitioning(particles, cell.r, (cut_dim+1)%2)

    return particles, cell

def plot_particle_cells(particles, cell:Cell, ax=None):

    x = [par.coord[0] for par in particles]
    y = [par.coord[1] for par in particles]

    if ax == None:
        print('no ax')
        fig, ax = plt.subplots()

    ax.scatter(x=x, y=y)
    ax.set_ylim(cell.c_low[1], cell.c_high[1])
    ax.set_xlim(cell.c_low[0], cell.c_high[0])

    def cell_rectange(cell:Cell):
        ax.add_patch(Rectangle((cell.c_low[0], cell.c_low[1]), cell.c_high[0]-cell.c_low[0], cell.c_high[1]-cell.c_low[1], edgecolor = 'k', fill=False))

        if cell.l != None:
            cell_rectange(cell.l)

        if cell.r != None:
            cell_rectange(cell.r)

    cell_rectange(cell)

    if ax == None:
        plt.show()

if __name__ == "__main__":

    particles = np.array([Particle([rng.random(), rng.random()]) for i in range(1000)])
    cell = Cell([0,0], [1,1], 0, len(particles))

    particles, cell = recursive_partitioning(particles, cell, 0)

    cell.print_all_cells()

    fig, ax = plt.subplots()
    plot_particle_cells(particles, cell, ax)
    plt.show()
