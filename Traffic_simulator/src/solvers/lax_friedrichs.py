import numpy as np


def lax_friedrichs_step(u_n, flux_n, dx, dt, boundary='periodic'):
    """
    Realiza un paso individual del metodo Lax-Friedrichs.

    Parametros:
        u_n (np.ndarray): Solucion en el paso temporal n
        flux_n (np.ndarray): Flujo evaluado en el paso temporal n
        dx (float): Espaciamiento espacial
        dt (float): Paso temporal
        boundary (str): Tipo de condicion de frontera ('periodic' u 'outflow')

    Retorna:
        np.ndarray: Solucion en el paso temporal n+1
    """
    nx = len(u_n)
    lambda_cfl = dt / dx
    u_new = np.zeros(nx)

    for i in range(nx):
        if boundary == 'periodic':
            i_left = (i - 1) % nx
            i_right = (i + 1) % nx
        else:  # outflow
            i_left = max(0, i - 1)
            i_right = min(nx - 1, i + 1)

        # Formula Lax-Friedrichs:
        # u_i^{n+1} = (u_{i-1}^n + u_{i+1}^n)/2 - (lambda/2)(f_{i+1}^n - f_{i-1}^n)
        u_new[i] = 0.5 * (u_n[i_left] + u_n[i_right]) - \
                   0.5 * lambda_cfl * (flux_n[i_right] - flux_n[i_left])

    return u_new


def lax_friedrichs_solve(u0, flux_func, x, t, boundary='periodic'):
    nx = len(x)
    nt = len(t)
    dx = x[1] - x[0]
    dt = t[1] - t[0] if len(t) > 1 else 0.01

    # Inicializar solucion
    u = np.zeros((nt, nx))
    u[0, :] = u0

    # Numero CFL
    lambda_cfl = dt / dx

    # Calcular velocidad maxima aproximada para estabilidad
    max_speed = np.max(np.abs(flux_func(u0) / (u0 + 1e-10)))

    # Escala de tiempo adaptativa si es necesario
    if lambda_cfl * max_speed > 1.0:
        print(f"Advertencia: numero CFL = {lambda_cfl * max_speed:.3f} > 1")

    # Iterar en tiempo
    for n in range(nt - 1):
        u_n = u[n, :]

        # Calcular flujos
        f = flux_func(u_n)

        # Aplicar metodo Lax-Friedrichs
        u_new = np.zeros(nx)

        for i in range(nx):
            if boundary == 'periodic':
                i_left = (i - 1) % nx
                i_right = (i + 1) % nx
            else:  # outflow
                i_left = max(0, i - 1)
                i_right = min(nx - 1, i + 1)

            # Formula Lax-Friedrichs:
            # u_i^{n+1} = (u_{i-1}^n + u_{i+1}^n)/2 - (lambda/2)(f_{i+1}^n - f_{i-1}^n)
            u_new[i] = 0.5 * (u_n[i_left] + u_n[i_right]) - \
                       0.5 * lambda_cfl * (f[i_right] - f[i_left])

        u[n + 1, :] = u_new

    return u


def adaptive_lax_friedrichs_solve(u0, flux_func, x, t_final, cfl_target=0.8, boundary='periodic'):
    """
    Resuelve una ecuacion de conservacion con paso temporal adaptativo.

    Parametros:
        u0 (np.ndarray): Condicion inicial
        flux_func (callable): Funcion de flujo
        x (np.ndarray): Malla espacial
        t_final (float): Tiempo final de simulacion
        cfl_target (float): Numero CFL objetivo para estabilidad
        boundary (str): Tipo de condicion de frontera

    Retorna:
        tuple: (u, t_array) donde u es la solucion y t_array es el vector de tiempos
    """
    nx = len(x)
    dx = x[1] - x[0]

    # Inicializar
    u_list = [u0.copy()]
    t_array = [0.0]
    t_current = 0.0
    u_current = u0.copy()

    max_iterations = 100000
    iteration = 0

    while t_current < t_final and iteration < max_iterations:
        iteration += 1

        # Calcular velocidad maxima
        f_current = flux_func(u_current)
        with np.errstate(divide='ignore', invalid='ignore'):
            speeds = np.abs(f_current / (u_current + 1e-10))
        max_speed = np.max(speeds)

        if max_speed < 1e-10:
            max_speed = 0.1

        # Calcular paso temporal segun CFL
        dt = min(cfl_target * dx / max_speed, t_final - t_current)
        dt = max(dt, 1e-6)

        # Realizar un paso
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
