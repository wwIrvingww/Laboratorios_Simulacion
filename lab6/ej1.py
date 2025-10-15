import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle

L = 10.0
Ntotal = 200
I0 = 5
vmax = 0.5
r = 0.3
beta = 0.5
gamma = 0.1
dt = 0.5
T = 200

np.random.seed(42)

class Particle:
    def __init__(self, x, y, vx, vy, state):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.state = state
        self.infection_time = 0
    
    def update_position(self):
        self.x += self.vx * dt
        self.y += self.vy * dt
        
        if self.x < 0 or self.x > L:
            self.vx *= -1
            self.x = np.clip(self.x, 0, L)
        if self.y < 0 or self.y > L:
            self.vy *= -1
            self.y = np.clip(self.y, 0, L)

particles = []
for i in range(Ntotal):
    x = np.random.uniform(0, L)
    y = np.random.uniform(0, L)
    angle = np.random.uniform(0, 2*np.pi)
    speed = np.random.uniform(0, vmax)
    vx = speed * np.cos(angle)
    vy = speed * np.sin(angle)
    
    state = 1 if i < I0 else 0
    particles.append(Particle(x, y, vx, vy, state))

S_history = []
I_history = []
R_history = []
time_history = []

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.set_xlim(0, L)
ax1.set_ylim(0, L)
ax1.set_aspect('equal')
ax1.set_xlabel('x')
ax1.set_ylabel('y')

scatter = ax1.scatter([], [], s=30)

ax2.set_xlim(0, T)
ax2.set_ylim(0, Ntotal)
ax2.set_xlabel('Tiempo')
ax2.set_ylabel('Número de individuos')
ax2.grid(True, alpha=0.3)

line_S, = ax2.plot([], [], 'b-', label='S', linewidth=2)
line_I, = ax2.plot([], [], 'r-', label='I', linewidth=2)
line_R, = ax2.plot([], [], 'g-', label='R', linewidth=2)
ax2.legend()

def init():
    scatter.set_offsets(np.empty((0, 2)))
    line_S.set_data([], [])
    line_I.set_data([], [])
    line_R.set_data([], [])
    return scatter, line_S, line_I, line_R

def animate(frame):
    t = frame * dt
    
    for p in particles:
        p.update_position()
        
        if p.state == 1:
            p.infection_time += dt
            if np.random.random() < gamma * dt:
                p.state = 2
    
    for i, p1 in enumerate(particles):
        if p1.state == 1:
            for j, p2 in enumerate(particles):
                if i != j and p2.state == 0:
                    dist = np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
                    if dist < r:
                        if np.random.random() < beta * dt:
                            p2.state = 1
    
    positions = np.array([[p.x, p.y] for p in particles])
    colors = ['blue' if p.state == 0 else 'red' if p.state == 1 else 'green' for p in particles]
    
    scatter.set_offsets(positions)
    scatter.set_color(colors)
    
    S = sum(1 for p in particles if p.state == 0)
    I = sum(1 for p in particles if p.state == 1)
    R = sum(1 for p in particles if p.state == 2)
    
    S_history.append(S)
    I_history.append(I)
    R_history.append(R)
    time_history.append(t)
    
    line_S.set_data(time_history, S_history)
    line_I.set_data(time_history, I_history)
    line_R.set_data(time_history, R_history)
    
    ax1.set_title(f't = {t:.1f} | S = {S} | I = {I} | R = {R}')
    
    return scatter, line_S, line_I, line_R

num_frames = int(T / dt)
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=num_frames, 
                              interval=100, blit=True, repeat=False)

plt.tight_layout()
plt.show()

import numpy as np

def run_simulation(initial_particles, L, r, beta, gamma, dt, T):
    """Simula la dinámica SIR por partículas y devuelve las series temporales."""
    # Clonar el conjunto inicial para no modificarlo
    particles = [Particle(p.x, p.y, p.vx, p.vy, p.state) for p in initial_particles]
    
    num_frames = int(T / dt)
    S_history, I_history, R_history, time_history = [], [], [], []
    
    for frame in range(num_frames):
        t = frame * dt

        # Actualizar posiciones y recuperaciones
        for p in particles:
            p.update_position()
            if p.state == 1:  # Infectado
                p.infection_time += dt
                if np.random.random() < gamma * dt:
                    p.state = 2  # Recuperado

        # Contagio entre partículas
        for i, p1 in enumerate(particles):
            if p1.state == 1:
                for j, p2 in enumerate(particles):
                    if i != j and p2.state == 0:
                        dist = np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
                        if dist < r and np.random.random() < beta * dt:
                            p2.state = 1

        # Contar poblaciones
        S = sum(1 for p in particles if p.state == 0)
        I = sum(1 for p in particles if p.state == 1)
        R = sum(1 for p in particles if p.state == 2)

        S_history.append(S)
        I_history.append(I)
        R_history.append(R)
        time_history.append(t)
    
    return np.array(S_history), np.array(I_history), np.array(R_history), np.array(time_history)
