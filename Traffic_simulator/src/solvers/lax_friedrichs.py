"""
Lax-Friedrichs solver
"""
import numpy as np


def lax_friedrichs_step(u_n, flux_n, dx, dt, boundary='periodic'):
    nx = len(u_n)
    lambda_cfl = dt / dx
    u_new = np.zeros(nx)

    for i in range(nx):
        if boundary == 'periodic':
            i_left = (i - 1) % nx
            i_right = (i + 1) % nx
        else:
            i_left = max(0, i - 1)
            i_right = min(nx - 1, i + 1)

        u_new[i] = 0.5 * (u_n[i_left] + u_n[i_right]) - \
                   0.5 * lambda_cfl * (flux_n[i_right] - flux_n[i_left])

    return u_new


def lax_friedrichs_solve(u0, flux_func, x, t, boundary='periodic'):
    nx = len(x)
    nt = len(t)
    dx = x[1] - x[0]
    dt = t[1] - t[0] if len(t) > 1 else 0.01

    u = np.zeros((nt, nx))
    u[0, :] = u0

    lambda_cfl = dt / dx
    max_speed = np.max(np.abs(flux_func(u0) / (u0 + 1e-10)))

    if lambda_cfl * max_speed > 1.0:
        print(f"Advertencia: numero CFL = {lambda_cfl * max_speed:.3f} > 1")

    for n in range(nt - 1):
        u_n = u[n, :]
        f = flux_func(u_n)
        u_new = np.zeros(nx)

        for i in range(nx):
            if boundary == 'periodic':
                i_left = (i - 1) % nx
                i_right = (i + 1) % nx
            else:
                i_left = max(0, i - 1)
                i_right = min(nx - 1, i + 1)

            u_new[i] = 0.5 * (u_n[i_left] + u_n[i_right]) - \
                       0.5 * lambda_cfl * (f[i_right] - f[i_left])

        u[n + 1, :] = u_new

    return u


def adaptive_lax_friedrichs_solve(u0, flux_func, x, t_final, cfl_target=0.8, boundary='periodic'):
    nx = len(x)
    dx = x[1] - x[0]

    u_list = [u0.copy()]
    t_array = [0.0]
    t_current = 0.0
    u_current = u0.copy()

    max_iterations = 100000
    iteration = 0

    while t_current < t_final and iteration < max_iterations:
        iteration += 1

        f_current = flux_func(u_current)
        with np.errstate(divide='ignore', invalid='ignore'):
            speeds = np.abs(f_current / (u_current + 1e-10))
        max_speed = np.max(speeds)

        if max_speed < 1e-10:
            max_speed = 0.1

        dt = min(cfl_target * dx / max_speed, t_final - t_current)
        dt = max(dt, 1e-6)

        lambda_cfl = dt / dx
        f = flux_func(u_current)

        u_new = np.zeros(nx)
        for i in range(nx):
            if boundary == 'periodic':
                i_left = (i - 1) % nx
                i_right = (i + 1) % nx
            else:
                i_left = max(0, i - 1)
                i_right = min(nx - 1, i + 1)

            u_new[i] = 0.5 * (u_current[i_left] + u_current[i_right]) - \
                      0.5 * lambda_cfl * (f[i_right] - f[i_left])

        u_current = u_new
        t_current += dt

        u_list.append(u_current.copy())
        t_array.append(t_current)

    return np.array(u_list), np.array(t_array)
