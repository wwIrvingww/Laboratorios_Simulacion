import numpy as np

V_MAX = 100.0
RHO_MAX = 150.0

L_ROAD = 10.0
T_FINAL = 1.0
DX = 0.1
DT = 0.001

def get_spatial_grid(L=L_ROAD, dx=DX):
    return np.arange(0, L + dx, dx)

def get_temporal_grid(T=T_FINAL, dt=DT):
    return np.arange(0, T + dt, dt)

def get_velocity(rho):
    return V_MAX * (1 - rho / RHO_MAX)

def get_flux(rho):
    return rho * get_velocity(rho)