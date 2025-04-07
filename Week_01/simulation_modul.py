import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

class Particle:
    def __init__(self, coordinate):
        self.coordinate = coordinate # Coordinate [x,y] tupple? 

    def __str__(self):
        return f"Particle:\n\tLocation: {self.coordinate}"
    
class Cell:
    def __init__(self, c_low, c_high, i_lower, i_upper):
        self.c_low = c_low # Coordinate [x,y]
        self.c_high = c_high # Coordinate [x,y]
        self.l = None # Cell
        self.r = None # Cell
        self.i_lower = i_lower # lower indices position (integer)
        self.i_upper = i_upper # upper indices position (integer)

    def __str__(self):
        return f"Cell:\n\tLocation: ({self.c_low}, {self.c_high})\n\tIndex: [{self.i_lower}:{self.i_upper}]"
    #what the point of this function?


    def print_all_cells(self):
        print(self)
        if self.l != None:
            self.l.print_all_cells()
        if self.r != None:
            self.r.print_all_cells()


def partitioning(particles, cell:Cell, cut_dim:int):
    # cut_dim is dimension in wich is cut, 0 -> x, 1 -> y


    # check if cell is empty
    if len(particles) == 0:
        print('Warning: Tried to partition an empty cell, this should not happen.')
        return particles, cell, 0, 0

    # check particel at low index, if bigger swap it with high index and decreas high index, if low leave it and increase low index
    c_mid = (cell.c_high[cut_dim] + cell.c_low[cut_dim])/2

    l = cell.i_lower
    h = cell.i_upper - 1

    for _ in range(cell.i_lower, cell.i_upper):
        if particles[l].coordinate[cut_dim] > c_mid:
            (particles[l], particles[h]) = (particles[h], particles[l])
            h = h - 1
        else:
            l = l + 1

    i_mid = l

    # calculate number of particles in left and right cell
    n_l = l - cell.i_lower
    n_r = cell.i_upper - l

    # plot_particles(particles[cell.i_lower:l])
    # plot_particles(particles[l:cell.i_upper])
    # plt.show()

    # calculate the new coordinates of the child cells
    l_low = cell.c_low.copy()
    l_high = cell.c_high.copy()
    l_high[cut_dim] = c_mid

    r_high = cell.c_high.copy()
    r_low = cell.c_low.copy()
    r_low[cut_dim] = c_mid

    # calculate the new indecies of the child cells
    l_lower = cell.i_lower
    l_upper = i_mid

    r_lower = i_mid
    r_upper = cell.i_upper

    # define the new l_cell and r_cell in the current cell
    cell.l = Cell(l_low, l_high, l_lower, l_upper)
    cell.r = Cell(r_low, r_high, r_lower, r_upper)

    # print(l_low)
    # print(l_high)

    # print(r_low)
    # print(r_high)

    return particles, cell, n_l, n_r

def recursive_partitioning(particles, cell, cut_dim, min_num=8):

    # run partitioning with current cell
    particles, cell, l_num, r_num = partitioning(particles, cell, cut_dim)
    
    # print('l_num: ', l_num)
    # print('r_num: ', r_num)

    #check if there are more than the minimum number of particles in a cell, and if so run the partitioning again
    if l_num > min_num:
        recursive_partitioning(particles, cell.l, (cut_dim+1)%2)

    if r_num > min_num:
        recursive_partitioning(particles, cell.r, (cut_dim+1)%2)

    return particles, cell

def plot_particles(particles):
    x = [par.coordinate[0] for par in particles]
    y = [par.coordinate[1] for par in particles]

    plt.scatter(x=x, y=y)
    plt.ylim(0,1)
    plt.xlim(0,1)

def plot_particles_color(particles):
    x = [par.coordinate[0] for par in particles]
    y = [par.coordinate[1] for par in particles]
    t = np.arange(len(x))

    plt.scatter(x=x, y=y, c=t)
    plt.ylim(0,1)
    plt.xlim(0,1)

def plot_particle_cells(particles, cell:Cell, ax=None):

    x = [par.coordinate[0] for par in particles]
    y = [par.coordinate[1] for par in particles]

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

### Week 2

from heapq import *

# Priority queue
class prioq:
    def __init__(self, k):
        self.heap = []
        sentinel = (-np.inf, None, np.array([0.0,0.0]))
        for i in range(k):
            heappush(self.heap, sentinel)

    def replace(self, dist2, particle, dr):
        # .... use heapreplace here
        heapreplace(self.heap, (dist2, particle, dr))

    def key(self):
        # .... define key here
        key = self.heap[0]
        return key[0]

def celldist2(self:Cell, r):
    """Calculates the squared minimum distance between a particle
    position and this node."""
    d1 = r - self.c_high
    d2 = self.c_low - r
    d1 = np.maximum(d1, d2)
    d1 = np.maximum(d1, np.zeros_like(d1))
    return d1.dot(d1)


# pq = prioq(k)

def neighbor_search_periodic(pq, root, particles, r, period):
    # walk the closest image first (at offset=[0, 0])
    for y in [0.0, -period[1], period[1]]:
        for x in [0.0, -period[0], period[0]]:
            rOffset = np.array([x, y])
            neighbor_search(pq, root, particles, r, rOffset)

def neighbor_search(pq:prioq, root:Cell, particles, r, rOffset):
    # Check if leave cell
    if (root.l == None) and (root.r == None):
        for parti in particles[root.i_lower:root.i_upper]:
            dist_parti = - np.sum(np.square(parti.coordinate + rOffset - r))
            if dist_parti > pq.key():
                dr = parti.coordinate + rOffset - r
                pq.replace(dist_parti, parti, dr)
    
    else:
        if - celldist2(root.l, r - rOffset) > pq.key():
            neighbor_search(pq, root.l, particles, r, rOffset)

        if - celldist2(root.r, r - rOffset) > pq.key():
            neighbor_search(pq, root.r, particles, r, rOffset)

def plot_pq(pq:prioq, r, period, ax, color = 'red'):
    neighbors = np.array([parti[1].coordinate for parti in pq.heap])
    # print(neighbors)
    ax.scatter(x=neighbors[:,0], y=neighbors[:,1], c=color)
    for y in [0.0, -period[1], period[1]]:
        for x in [0.0, -period[0], period[0]]:
            rOffset = np.array([x, y])
            r2 = r - rOffset
            ax.add_patch(Circle(xy=(r2[0], r2[1]), radius= np.sqrt(-pq.key()), edgecolor = 'k', fill=False))

