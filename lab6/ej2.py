# ca_sir_ca_part2.py
"""
Parte 2 — Simulación SIR mediante autómata celular (una sola ejecución).
Genera:
 - ca_simulation.gif, ca_simulation.mp4
 - sir_curves.png   (S/I/R de la corrida única)
 - snapshot_t{t}.png para los tiempos solicitados

Requisitos: numpy, matplotlib, imageio. (scipy optional para convolución).
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
import math
from numpy.random import Generator, PCG64
try:
    from scipy.signal import convolve2d
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# Estados
SUSCEPTIBLE = 0
INFECTED = 1
RECOVERED = 2

STATE_COLORS = {
    SUSCEPTIBLE: "#1f77b4",
    INFECTED: "#d62728",
    RECOVERED: "#2ca02c",
}


def generate_initial_map(M: int, N: int, I0: int, seed_init: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed_init)
    grid = np.full((M, N), SUSCEPTIBLE, dtype=np.uint8)
    total = M * N
    I0 = min(I0, total)
    chosen = rng.choice(total, size=I0, replace=False)
    grid.reshape(-1)[chosen] = INFECTED
    return grid


def make_circular_kernel(r: float) -> np.ndarray:
    rad = math.ceil(r)
    size = 2 * rad + 1
    ys, xs = np.ogrid[-rad:rad+1, -rad:rad+1]
    dist = np.sqrt(xs**2 + ys**2)
    mask = (dist <= r).astype(np.float64)
    mask[rad, rad] = 0.0
    return mask


def _convolve(grid_float: np.ndarray, kernel: np.ndarray, boundary: str = "periodic") -> np.ndarray:
    if _HAS_SCIPY:
        bc = 'wrap' if boundary == "periodic" else 'fill'
        return convolve2d(grid_float, kernel, mode='same', boundary=bc, fillvalue=0.0)
    else:
        # fallback simple: usar FFT para contorno periódico, pad+fft for fixed
        if boundary == "periodic":
            M, N = grid_float.shape
            kshape = np.zeros_like(grid_float, dtype=np.float64)
            kr, kc = kernel.shape
            pr = (M - kr) // 2
            pc = (N - kc) // 2
            kshape[pr:pr+kr, pc:pc+kc] = kernel
            kshape = np.fft.ifftshift(kshape)
            fftg = np.fft.rfft2(grid_float)
            fftk = np.fft.rfft2(kshape)
            conv = np.fft.irfft2(fftg * fftk, s=grid_float.shape)
            return np.real(conv)
        else:
            # simple but menos eficiente para fixed boundary
            M, N = grid_float.shape
            kr, kc = kernel.shape
            outshape = (M + kr - 1, N + kc - 1)
            fg = np.fft.rfft2(grid_float, s=outshape)
            fk = np.fft.rfft2(kernel, s=outshape)
            conv_full = np.fft.irfft2(fg * fk, s=outshape)
            start_r = (kr - 1) // 2
            start_c = (kc - 1) // 2
            return conv_full[start_r:start_r+M, start_c:start_c+N]


def step_ca(grid: np.ndarray, kernel: np.ndarray, beta: float, gamma: float, dt: float,
            rng: Generator, boundary: str = "periodic") -> np.ndarray:
    infected = (grid == INFECTED).astype(np.float64)
    neighbor_inf_sum = _convolve(infected, kernel, boundary=boundary)
    kernel_sum = kernel.sum()
    if boundary == "fixed" and not _HAS_SCIPY:
        ones = np.ones_like(grid, dtype=np.float64)
        neigh_count = _convolve(ones, kernel, boundary=boundary)
        neigh_count = np.maximum(neigh_count, 1.0)
        p_vec = neighbor_inf_sum / neigh_count
    else:
        if kernel_sum <= 0:
            raise ValueError("Kernel vacío (aumenta r).")
        p_vec = neighbor_inf_sum / kernel_sum

    p_inf = 1.0 - np.exp(-beta * p_vec * dt)
    p_rec = 1.0 - np.exp(-gamma * dt)

    U = rng.random(size=grid.shape)
    V = rng.random(size=grid.shape)

    new_grid = grid.copy()
    sus_mask = (grid == SUSCEPTIBLE)
    infect_mask = sus_mask & (U < p_inf)
    new_grid[infect_mask] = INFECTED

    inf_mask = (grid == INFECTED)
    rec_mask = inf_mask & (V < p_rec)
    new_grid[rec_mask] = RECOVERED

    return new_grid


def run_single(grid0: np.ndarray, kernel: np.ndarray, beta: float, gamma: float, dt: float,
               n_steps: int, rng: Generator, boundary: str = "periodic", record_frames: bool = True):
    grid = grid0.copy()
    timeseries = np.zeros((n_steps + 1, 3), dtype=np.int32)
    def counts(g):
        return (np.sum(g == SUSCEPTIBLE), np.sum(g == INFECTED), np.sum(g == RECOVERED))
    timeseries[0] = counts(grid)
    frames = [grid.copy()] if record_frames else []
    for t in range(1, n_steps + 1):
        grid = step_ca(grid, kernel, beta, gamma, dt, rng, boundary=boundary)
        timeseries[t] = counts(grid)
        if record_frames:
            frames.append(grid.copy())
    return timeseries, frames


def save_animation(frames, filename_gif, filename_mp4, fps=10):
    if not frames:
        print("Sin frames para animar.")
        return
    images = []
    for g in frames:
        h, w = g.shape
        img = np.zeros((h, w, 3), dtype=np.uint8)
        for state, hexcol in STATE_COLORS.items():
            rgb = tuple(int(hexcol.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            img[g == state] = rgb
        images.append(img)
    try:
        imageio.mimsave(filename_gif, images, fps=fps)
        print("Gif guardado:", filename_gif)
    except Exception as e:
        print("No se pudo guardar GIF:", e)
    try:
        writer = imageio.get_writer(filename_mp4, fps=fps)
        for im in images:
            writer.append_data(im)
        writer.close()
        print("MP4 guardado:", filename_mp4)
    except Exception as e:
        print("No se pudo guardar MP4:", e)


def save_snapshots(frames, snapshot_times, out_dir="."):
    if not frames:
        return
    n_steps = len(frames) - 1
    for t in snapshot_times:
        idx = int(round(t))
        idx = max(0, min(n_steps, idx))
        frame = frames[idx]
        h, w = frame.shape
        img = np.zeros((h, w, 3), dtype=np.uint8)
        for state, hexcol in STATE_COLORS.items():
            rgb = tuple(int(hexcol.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            img[frame == state] = rgb
        out_name = os.path.join(out_dir, f"snapshot_t{idx}.png")
        imageio.imwrite(out_name, img)
        print("Snapshot guardado:", out_name)


def plot_single(timeseries, times, out_png="sir_curves.png"):
    Ntot = timeseries[0].sum()
    S = timeseries[:, 0] / Ntot
    I = timeseries[:, 1] / Ntot
    R = timeseries[:, 2] / Ntot
    plt.figure(figsize=(8,5))
    plt.plot(times, S, label="S", color=STATE_COLORS[SUSCEPTIBLE])
    plt.plot(times, I, label="I", color=STATE_COLORS[INFECTED])
    plt.plot(times, R, label="R", color=STATE_COLORS[RECOVERED])
    plt.xlabel("Tiempo")
    plt.ylabel("Fracción población")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print("Curvas guardadas:", out_png)


def main_part2(M, N, I0, T, dt, r, beta, gamma, seed_init, boundary, snapshot_times, out_dir="."):
    os.makedirs(out_dir, exist_ok=True)
    initial = generate_initial_map(M, N, I0, seed_init=seed_init)
    kernel = make_circular_kernel(r)
    rng = Generator(PCG64(seed_init if seed_init is not None else 12345))
    n_steps = int(T)
    times = np.arange(0, n_steps+1) * dt
    timeseries, frames = run_single(initial, kernel, beta, gamma, dt, n_steps, rng, boundary=boundary, record_frames=True)
    # guardar outputs solicitados por Parte 2
    gif_path = os.path.join(out_dir, "ca_simulation.gif")
    mp4_path = os.path.join(out_dir, "ca_simulation.mp4")
    save_animation(frames, gif_path, mp4_path, fps=10)
    save_snapshots(frames, snapshot_times, out_dir=out_dir)
    plot_single(timeseries, times, out_png=os.path.join(out_dir, "sir_curves.png"))
    print("Parte 2 completada. Archivos en:", out_dir)


if __name__ == "__main__":
    # Parámetros por defecto (parte 2)
    M, N = 200, 200
    I0 = 100
    T = 200
    dt = 1.0
    r = 1.5
    beta = 0.6
    gamma = 0.1
    seed_init = 42
    boundary = "periodic"
    snapshot_times = [0, int(T*0.25), int(T*0.5), int(T*0.75), int(T)]
    out_dir = "ca_part2_outputs"
    main_part2(M, N, I0, T, dt, r, beta, gamma, seed_init, boundary, snapshot_times, out_dir=out_dir)
