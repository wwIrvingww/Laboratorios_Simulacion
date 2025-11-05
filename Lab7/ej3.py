import os
import csv
import math
import time
from typing import List, Dict, Any, Iterable, Tuple

import numpy as np

try:
    import sts  # type: ignore  # módulo funcional con las pruebas NIST expuesto por sts-pylib
except ImportError as e:
    raise SystemExit(
        "No se pudo importar 'sts' (API de NIST SP 800-22).\n"
        "Instale el paquete: pip install sts-pylib\n"
        f"Detalle: {e}"
    ) from e


# ---------------------------
# Generadores de bits (1e6)
# ---------------------------
class LCG:
    """Generador Congruencial Lineal (Park-Miller MINSTD por defecto)."""

    def __init__(self, m: int, a: int, c: int, seed: int):
        self.m = m
        self.a = a
        self.c = c
        self.current = seed % m

    def next(self) -> int:
        self.current = (self.a * self.current + self.c) % self.m
        return self.current

    def uniform01(self) -> float:
        return self.next() / self.m


def bits_from_uniform_threshold(u: np.ndarray) -> np.ndarray:
    """Mapea U[0,1) -> bits por umbral: 1 si u>=0.5, 0 en caso contrario."""
    return (u >= 0.5).astype(np.uint8)


def generate_bits_lcg(n_bits: int, seed: int = 42, variant: str = "park-miller") -> np.ndarray:
    if variant == "park-miller":
        # MINSTD
        m, a, c = (2 ** 31 - 1), 48271, 0
    elif variant == "numerical-recipes":
        m, a, c = (2 ** 32), 1664525, 1013904223
    else:
        raise ValueError("variant debe ser 'park-miller' o 'numerical-recipes'")

    lcg = LCG(m, a, c, seed)
    # Generar en bloques para rendimiento
    chunk = 100_000
    bits = np.empty(n_bits, dtype=np.uint8)
    idx = 0
    while idx < n_bits:
        k = min(chunk, n_bits - idx)
        u = np.empty(k, dtype=np.float64)
        for i in range(k):
            u[i] = lcg.uniform01()
        bits[idx: idx + k] = bits_from_uniform_threshold(u)
        idx += k
    return bits


def generate_bits_mt(n_bits: int, seed: int = 42) -> np.ndarray:
    # Usar explícitamente MT19937 para asegurar Mersenne Twister
    bitgen = np.random.MT19937(seed)
    rng = np.random.Generator(bitgen)
    u = rng.random(n_bits, dtype=np.float64)
    return bits_from_uniform_threshold(u)


# ---------------------------
# Ejecutor de pruebas NIST
# ---------------------------
ALPHA = 0.01  # Umbral típico de NIST STS


def _pass(p: float) -> bool:
    try:
        return (p is not None) and (not math.isnan(p)) and (p >= ALPHA)
    except (TypeError, ValueError):
        return False


def run_nist_tests(epsilon: Iterable[int]) -> List[Dict[str, Any]]:
    eps = list(int(b) for b in epsilon)  # sts espera lista de ints 0/1

    out_rows: List[Dict[str, Any]] = []

    def add(name: str, pval: Any, detail: str = ""):
        if isinstance(pval, (list, tuple)):
            for j, pv in enumerate(pval):
                out_rows.append({
                    "Test": name,
                    "Detail": f"{detail}#{j+1}" if detail else f"subtest#{j+1}",
                    "p_value": float(pv) if pv is not None else float("nan"),
                    "Pass": _pass(pv),
                })
        else:
            out_rows.append({
                "Test": name,
                "Detail": detail,
                "p_value": float(pval) if pval is not None else float("nan"),
                "Pass": _pass(pval),
            })

    # 1. Frequency (Monobit)
    add("Frequency (Monobit)", sts.frequency(eps))

    # 2. Block Frequency (usar M=128 por defecto común)
    add("Block Frequency (M=128)", sts.block_frequency(eps, 128))

    # 3. Runs
    add("Runs", sts.runs(eps))

    # 4. Longest Run of Ones in a Block
    add("Longest Run of Ones", sts.longest_run_of_ones(eps))

    # 5. Binary Matrix Rank
    add("Binary Matrix Rank", sts.rank(eps))

    # 6. Discrete Fourier Transform (Spectral)
    add("Discrete Fourier Transform", sts.discrete_fourier_transform(eps))

    # 7. Non-overlapping Template Matchings (NO DISPONIBLE en esta copia: NotImplemented)
    try:
        add("Non-overlapping Template Matchings (m=9)", sts.non_overlapping_template_matchings(eps, 9))
    except NotImplementedError:
        add("Non-overlapping Template Matchings (m=9)", float("nan"), detail="N/A (NotImplemented)")

    # 8. Overlapping Template Matchings (m=9)
    add("Overlapping Template Matchings (m=9)", sts.overlapping_template_matchings(eps, 9))

    # 9. Maurer's Universal
    add("Maurer's Universal", sts.universal(eps))

    # 10. Linear Complexity (M=500)
    add("Linear Complexity (M=500)", sts.linear_complexity(eps, 500))

    # 11. Serial (m=16) devuelve dos p-valores
    serial_pvals = sts.serial(eps, 16)
    add("Serial (m=16)", serial_pvals)

    # 12. Approximate Entropy (m=10)
    add("Approximate Entropy (m=10)", sts.approximate_entropy(eps, 10))

    # 13. Cumulative Sums (Forward y Backward)
    add("Cumulative Sums (Forward)", sts.cumulative_sums(eps, reverse=False))
    add("Cumulative Sums (Backward)", sts.cumulative_sums(eps, reverse=True))

    # 14. Random Excursions (8 p-valores)
    add("Random Excursions", sts.random_excursions(eps))

    # 15. Random Excursions Variant (18 p-valores)
    add("Random Excursions Variant", sts.random_excursions_variant(eps))

    return out_rows


def compare_generators(n_bits: int = 1_000_000, seed: int = 42) -> List[Dict[str, Any]]:
    datasets: List[Tuple[str, np.ndarray]] = []

    print(f"Generando {n_bits:,} bits con LCG (Park-Miller)...")
    t0 = time.time()
    bits_lcg_pm = generate_bits_lcg(n_bits, seed=seed, variant="park-miller")
    print(f"  listo en {time.time()-t0:.2f}s")
    datasets.append(("LCG (Park-Miller)", bits_lcg_pm))

    print(f"Generando {n_bits:,} bits con Mersenne Twister (MT19937)...")
    t0 = time.time()
    bits_mt = generate_bits_mt(n_bits, seed=seed)
    print(f"  listo en {time.time()-t0:.2f}s")
    datasets.append(("Mersenne Twister (MT19937)", bits_mt))

    all_rows: List[Dict[str, Any]] = []
    for name, bits in datasets:
        print(f"Ejecutando NIST SP 800-22 sobre: {name} (α={ALPHA}) ...")
        t0 = time.time()
        test_rows = run_nist_tests(bits)
        for r in test_rows:
            r["Generator"] = name
        print(f"  pruebas completadas en {time.time()-t0:.2f}s; passed: {sum(r['Pass'] for r in test_rows)}/{len(test_rows)}")
        all_rows.extend(test_rows)

    return all_rows


def save_results(records: List[Dict[str, Any]], out_csv: str) -> None:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    if not records:
        print("No hay resultados para guardar.")
        return
    fields = ["Generator", "Test", "Detail", "p_value", "Pass"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k, "") for k in fields})
    print(f"Resultados guardados en: {out_csv}")


def print_brief_summary(records: List[Dict[str, Any]]) -> None:
    if not records:
        return
    gens = sorted(set(r["Generator"] for r in records))
    print("\nResumen por generador (α = {:.2g}):".format(ALPHA))
    for g in gens:
        gr = [r for r in records if r["Generator"] == g]
        passed = sum(r["Pass"] for r in gr)
        print(f"  - {g}: {passed}/{len(gr)} pruebas/subpruebas pasan")


if __name__ == "__main__":
    N_BITS = 1_000_000
    SEED = 42

    results = compare_generators(n_bits=N_BITS, seed=SEED)
    save_path = os.path.join(os.path.dirname(__file__), "nist_results.csv")
    save_results(results, save_path)
    print_brief_summary(results)
    print("\nConcluya: compare el desempeño de los generadores a partir de los p-values y la tasa de aprobación.")
