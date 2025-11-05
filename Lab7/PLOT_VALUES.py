#!/usr/bin/env python3
"""Genera gráficas a partir de `generator_comparison.csv` de forma robusta.

Uso: python3 plot_pvalues.py
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = os.path.dirname(__file__)
CSV_IN = os.path.join(ROOT, 'generator_comparison.csv')
IMG_DIR = os.path.join(ROOT, 'img')
os.makedirs(IMG_DIR, exist_ok=True)


def main():
    df = pd.read_csv(CSV_IN)
    # detect available p_value columns
    pcols = [c for c in df.columns if c.startswith('p_value_')]
    if len(pcols) == 0:
        print('No se encontraron columnas p_value_ en', CSV_IN)
        return
    # Melt to long form
    vals = df[pcols]
    dfm = vals.melt(var_name='generator', value_name='p_value')
    dfm['generator'] = dfm['generator'].astype(str).str.replace('p_value_', '')
    dfm = dfm.dropna(subset=['p_value'])

    out1 = os.path.join(IMG_DIR, 'boxplot_pvalues.png')
    if dfm.empty:
        plt.figure(figsize=(6, 3))
        plt.text(0.5, 0.5, 'No p-value data available', ha='center', va='center')
        plt.axis('off')
        plt.savefig(out1, bbox_inches='tight', dpi=150)
        plt.close()
        print('Saved placeholder', out1)
    else:
        order = sorted(dfm['generator'].unique())
        plt.figure(figsize=(8, 5))
        sns.boxplot(x='generator', y='p_value', data=dfm, order=order)
        sns.stripplot(x='generator', y='p_value', data=dfm, color='k', alpha=0.4, order=order)
        plt.ylim(-0.05, 1.05)
        plt.title('Distribución de p-values por generador')
        plt.savefig(out1, bbox_inches='tight', dpi=150)
        plt.close()
        print('Saved', out1)

    # Scatter paired — try to find matching pairs in the CSV
    if len(pcols) >= 2:
        g1 = pcols[0]
        g2 = pcols[1]
        mask = df[[g1, g2]].notna().all(axis=1)
        out2 = os.path.join(IMG_DIR, 'scatter_pvalues.png')
        if mask.sum() == 0:
            plt.figure(figsize=(6, 3))
            plt.text(0.5, 0.5, 'No paired p-value data available', ha='center', va='center')
            plt.axis('off')
            plt.savefig(out2, bbox_inches='tight', dpi=150)
            plt.close()
            print('Saved placeholder', out2)
        else:
            plt.figure(figsize=(6, 6))
            plt.scatter(df.loc[mask, g1], df.loc[mask, g2])
            plt.plot([0, 1], [0, 1], '--', color='gray')
            plt.xlabel(g1.replace('p_value_', ''))
            plt.ylabel(g2.replace('p_value_', ''))
            plt.title('P-values pareados: %s vs %s' % (g1.replace('p_value_', ''), g2.replace('p_value_', '')))
            plt.savefig(out2, bbox_inches='tight', dpi=150)
            plt.close()
            print('Saved', out2)


if __name__ == '__main__':
    main()
