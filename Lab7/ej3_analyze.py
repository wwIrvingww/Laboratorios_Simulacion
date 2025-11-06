"""Analiza los resultados de NIST guardados en Lab7/nist_results.csv
Genera gráficas en Lab7/ej3/img:
 - pass_rate_by_generator.png : tasa de aprobación por generador
 - pvalue_heatmap.png         : heatmap de p-values (tests x generadores)
 - pvalue_boxplot.png         : distribución de p-values por generador

Uso:
    python Lab7/ej3_analyze.py

Dependencias: pandas, matplotlib, seaborn, numpy
"""
import os
import sys
import csv
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


ROOT = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(ROOT, "nist_results.csv")
OUT_DIR = os.path.join(ROOT, "ej3", "img")
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR, exist_ok=True)


def load_results(csv_path: str) -> pd.DataFrame:
    # Leer CSV y normalizar tipos
    df = pd.read_csv(csv_path)
    # Asegurar columnas esperadas
    expected = {"Generator", "Test", "Detail", "p_value", "Pass"}
    if not expected.issubset(df.columns):
        raise RuntimeError(f"CSV no contiene las columnas esperadas: {expected}")
    # Convertir p_value a numeric (nan posible)
    df["p_value"] = pd.to_numeric(df["p_value"], errors="coerce")
    df["Pass"] = df["Pass"].astype(bool)
    return df


def plot_pass_rate(df: pd.DataFrame, out_dir: str) -> str:
    # Calcular pass rate por generador
    grp = df.groupby("Generator").agg(total=("Pass", "size"), passed=("Pass", "sum"))
    grp["pass_rate"] = grp["passed"] / grp["total"]
    grp = grp.sort_values("pass_rate", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=grp.index, y=grp["pass_rate"], palette="viridis", ax=ax)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Pass rate (α = 0.01)")
    ax.set_xlabel("")
    ax.set_title("Tasa de aprobación por generador (NIST SP 800-22)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
    plt.tight_layout()

    out = os.path.join(out_dir, "pass_rate_by_generator.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_pvalue_heatmap(df: pd.DataFrame, out_dir: str) -> str:
    # Tomar p_values promedio por Test x Generator (cuando hay subtests, se muestran como filas separadas)
    # Para la visualización, pivotamos Test (posiblemente agregando Detail) vs Generator
    df2 = df.copy()
    # crear etiqueta de prueba única por Test+Detail
    df2["TestLabel"] = df2.apply(lambda r: r["Test"] if pd.isna(r["Detail"]) or r["Detail"] == "" else f"{r['Test']} {r['Detail']}", axis=1)

    pivot = df2.pivot_table(index="TestLabel", columns="Generator", values="p_value", aggfunc="mean")
    # Ordenar tests por nombre para estabilidad
    pivot = pivot.sort_index()

    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(pivot.index))))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="coolwarm", cbar_kws={"label": "p-value"}, ax=ax)
    ax.set_title("Heatmap de p-values (Test x Generador)")
    plt.tight_layout()

    out = os.path.join(out_dir, "pvalue_heatmap.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_pvalue_boxplot(df: pd.DataFrame, out_dir: str) -> str:
    # Boxplot de p_values por generador
    df_clean = df.dropna(subset=["p_value"]).copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x="Generator", y="p_value", data=df_clean, palette="Set2", ax=ax)
    sns.stripplot(x="Generator", y="p_value", data=df_clean, color="k", size=3, alpha=0.4, ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Distribución de p-values por generador")
    ax.set_ylabel("p-value")
    ax.set_xlabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
    plt.tight_layout()

    out = os.path.join(out_dir, "pvalue_boxplot.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main():
    if not os.path.exists(CSV_PATH):
        print(f"Archivo de resultados no encontrado: {CSV_PATH}")
        sys.exit(1)

    df = load_results(CSV_PATH)
    print("Cargando datos:", CSV_PATH)
    out1 = plot_pass_rate(df, OUT_DIR)
    out2 = plot_pvalue_heatmap(df, OUT_DIR)
    out3 = plot_pvalue_boxplot(df, OUT_DIR)

    print("Generados:")
    for p in (out1, out2, out3):
        print(" -", p)


if __name__ == "__main__":
    main()
