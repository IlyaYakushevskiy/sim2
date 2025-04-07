import numpy as np
import numpy.random as rng
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from heapq import *
import matplotlib.cm as cmap
import matplotlib.colors as colors
import random
from matplotlib.animation import FuncAnimation




class Particle: 
    def __init__(self, coord, vel=None):
        self.coord = coord
        self.vel = vel if vel is not None else np.array([0.0, 0.0])
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
        sentinel = (-np.inf, None, np.array([0.0, 0.0]), np.array([0.0, 0.0]))  # (dist^2, particle, dr, velocity)
        for _ in range(k):
            heappush(self.heap, sentinel)

    def replace(self, dist2, particle, dr, velocity):
        heapreplace(self.heap, (dist2, particle, dr, velocity))

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
                pq.replace(dist2, p, curr_dist, p.vel)

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


###WEEK 3 

# def top_hat_kernel(r, h):
#     norm_factor = 1 / (np.pi * h**2)
#     return norm_factor if r < h else 0


def monaghan_kernel(r, h ):
    # if r == 0:
    #     print("r == 0")
    q = r / h
    norm_factor = 10 / (7 * np.pi * h**2)
    if q < 0.5:
        return norm_factor*(1 - 6 * q**2 + 6 * q**3)
    elif q <= 1:
        return norm_factor *(2 * (1 - q)**3)
    else:
        return 0 #why is this being triggered? 
    
def grad_monaghan_kernel(dr, h):
    r_j = np.linalg.norm(dr)  
    if r_j == 0:
        return np.array([0.0, 0.0])  

    q = r_j / h
    norm_factor = 10 / (7 * np.pi * h**2)

    if q < 0.5:
        return (12 * q + 18 * q**2) * norm_factor * (dr / r_j)  
    elif q < 1:
        return (-6 * (1 - q)**2) * norm_factor * (dr / r_j)  
    else:
        return np.array([0.0, 0.0]) 
    
def density(pq, m = 1, kernel="monaghan"):
    rho = 0

    for p_j in pq.heap:
        #print(p_j)
        r_j = np.linalg.norm(p_j[2])

        if r_j == 0.0 :
            continue  # Skip sentinel values
    
        h = np.linalg.norm(pq.heap[0][2]) 
        rho += m * monaghan_kernel(r_j, h)

    if rho <= 0:
        print(f"Error: Density is {rho} for particle {p_j[1].coord}")

    return rho

def pressure(rho, u):
    return 6 * rho * u 
   


def a(pq, p_i, rho_i, u = 0,  m=1):
    a_i = np.array([0.0, 0.0])

    
    for dist2, p_j, dr, v_j in pq.heap:
        if p_j is None:
            continue  

        rho_j = density(pq, m, kernel="monaghan")  

        if rho_j == 0 or rho_i == 0:
            continue

        p_j_val = pressure(rho_j, u)  

        h = np.linalg.norm(pq.heap[0][2]) 

        grad_W = grad_monaghan_kernel(dr, h)  

        a_i += ((p_j_val / rho_j**2) + (p_i / rho_i**2)) * grad_W 
        
    return -a_i


def u_dot(pq, p_i, rho_i,  v_i, m=1): 

    u_dot_i = 0
    
    for _, p_j, dr, v_j in pq.heap:
        if p_j is None: #should never happen, density is always positive ! 
            continue  

        if rho_i == 0:
            continue

        h = np.linalg.norm(pq.heap[0][2]) 
        grad_W = grad_monaghan_kernel(dr, h)  

        v_ij = v_i - v_j  

        u_dot_i += m * (p_i / rho_i**2) * np.dot(grad_W, v_ij)  

    return -u_dot_i

def speed_of_sound(u_pred):
    return np.sqrt(7*(7-1) * u_pred)
    
#be calculated for each particle and it's n neighbours from pq
def sph_update(pq, r, u, v, rho0, dt, m=1):

    rho = density(pq, m=m, kernel="monaghan") 
    p = pressure(rho, u)

    
    a_val = a(pq, p, rho, m) 
    u_dot_val = u_dot(pq, p, rho, v, m)

    # DRIFT1
    r_half = (r + (v * dt / 2)) % 1
    v_pred = v + (a_val * dt / 2)
    u_pred = u + (u_dot_val * dt / 2)

    # KICK
    a_half = a(pq, p, rho, m)
    u_dot_half = u_dot(pq, p, rho, v_pred, m)


    v_new = v + a_half * dt
    u_new = u + u_dot_half * dt

    #DRIFT2 
    r_new = (r_half + v_new * dt/2) % 1

    rho_new = density(pq, m=m, kernel="monaghan")
    c = speed_of_sound(u_pred)

    p_new = pressure(rho_new,u_new )

    return {"r": r_new, "v": v_new, "u": u_new, "rho": rho_new, "p": p_new}


def update(frame):
    global particles, velocities, rhos
    for i, particle in enumerate(particles):
        
        pq = prioq(k)
        neighbor_search_periodic(pq, cell, particles, particle.coord, np.array([1, 1]))
        
        rhos[i] = density(pq, m=1, kernel="monaghan")
        
        sph = sph_update(pq, particle.coord, u=0, v=velocities[i], rho0=1, dt=0.006)
        particle.coord = (sph["r"]) % 1


    coords = np.array([p.coord for p in particles])
    sc.set_offsets(coords)
    sc.set_array(rhos)
    return sc,



if __name__ == "__main__":
    k = 32
    particles = np.array([Particle([np.random.random(), np.random.random()]) for _ in range(100)])
    cell = Cell([0, 0], [1, 1], 0, len(particles))
    particles, cell = recursive_partitioning(particles, cell, 0)

    rhos = np.zeros(len(particles))
    velocities = np.random.uniform(-1, 1, (len(particles), 2)) * 0.01

    fig, ax = plt.subplots(figsize=(8, 8))
    sc = ax.scatter([p.coord[0] for p in particles], [p.coord[1] for p in particles], c=rhos, cmap="viridis")

    ani = FuncAnimation(fig, update, interval=1, blit=False, save_count=100)
    ani.save("sph.mp4", writer="ffmpeg", fps=10)