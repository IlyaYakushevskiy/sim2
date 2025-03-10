import numpy as np
import numpy.random as rng
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from heapq import *
from tree_build import *

"""pasting task 1 to submit 1 file  """
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
"""end of task 1"""
class prioq:
    def __init__(self, k):
        self.heap = []
        sentinel = (-np.inf, None, np.array([0.0, 0.0]))
        for i in range(k):
            heappush(self.heap, sentinel)

    def replace(self, dist2, particle, dr):
        heapreplace(self.heap, (dist2, particle, dr))

    def key(self):
        return self.heap[0][0]

def celldist2(cell, r):
    d1 = r - cell.c_high
    d2 = cell.c_low - r
    d1 = np.maximum(d1, d2)
    d1 = np.maximum(d1, np.zeros_like(d1))
    return np.dot(d1, d1)

def neighbor_search(pq, root, particles, r, r_o):
    if root.l is None and root.r is None:
        for p in particles[root.i_lower:root.i_upper]:
            dist2 = -np.sum(np.square(p.coord + r_o - r))
            if dist2 > pq.key():
                curr_dist = p.coord + r_o - r
                pq.replace(dist2, p, curr_dist)
    else:
        if -celldist2(root.l, r - r_o) > pq.key():
            neighbor_search(pq, root.l, particles, r, r_o)

        if -celldist2(root.r, r - r_o) > pq.key():
            neighbor_search(pq, root.r, particles, r, r_o)

def neighbor_search_periodic(pq, root, particles, r, period):
    for y in [0.0, -period[1], period[1]]:
        for x in [0.0, -period[0], period[0]]:
            r_offset = np.array([x, y])
            neighbor_search(pq, root, particles, r, r_offset)

def plot_NNS(pq, r, period, ax, color='red'):
    neighbors = np.array([p[1].coord for p in pq.heap if p[1] is not None])
    ax.scatter(neighbors[:, 0], neighbors[:, 1], c=color, edgecolor='black', s=50)

    radius = np.sqrt(-pq.key())
    for y in [0.0, -period[1], period[1]]:
        for x in [0.0, -period[0], period[0]]:
            r_offset = np.array([x, y])
            r2 = r - r_offset
            ax.add_patch(Circle(xy=(r2[0], r2[1]), radius=radius, edgecolor=color, fill=False, linestyle='dashed'))

if __name__ == "__main__":
    k = 20
    particles = np.array([Particle([rng.random(), rng.random()]) for _ in range(1000)])
    cell = Cell([0, 0], [1, 1], 0, len(particles))
    particles, cell = recursive_partitioning(particles, cell, 0)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter([p.coord[0] for p in particles], [p.coord[1] for p in particles], color='blue', s=10)

    center_p = Particle([0.5, 0.5])
    boundary_p = Particle([0.9, 0.9])

    pq_center = prioq(k)
    neighbor_search_periodic(pq_center, cell, particles, center_p.coord, np.array([1, 1]))

    pq_boundary = prioq(k)
    neighbor_search_periodic(pq_boundary, cell, particles, boundary_p.coord, np.array([1, 1]))

    ax.scatter(center_p.coord[0], center_p.coord[1], color='red')
    ax.scatter(boundary_p.coord[0], boundary_p.coord[1], color='orange')

    plot_NNS(pq_center, center_p.coord, np.array([1, 1]), ax, color='green')
    plot_NNS(pq_boundary, boundary_p.coord, np.array([1, 1]), ax, color='purple')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.legend()
    #plt.show()
    plt.savefig('NNSearch.png')


