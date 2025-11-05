#!/usr/bin/env python3
"""Analiza `nist_results.csv` comparando generadores.

Produce:
- `generator_comparison.csv` en el mismo directorio Lab7
- Gráficas en `img/` (boxplot y scatter) dentro de Lab7

Uso: python3 analyze_generators.py
"""
import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


ROOT = os.path.dirname(__file__)
CSV_IN = os.path.join(ROOT, "nist_results.csv")
OUT_CSV = os.path.join(ROOT, "generator_comparison.csv")
IMG_DIR = os.path.join(ROOT, "img")
os.makedirs(IMG_DIR, exist_ok=True)


def load_data(path):
    df = pd.read_csv(path)
    # normalize column names
    df.columns = [c.strip() for c in df.columns]
    # Convert p_value to numeric (coerce NaN)
    df['p_value'] = pd.to_numeric(df['p_value'], errors='coerce')
    # Ensure Pass is boolean
    if df['Pass'].dtype != bool:
        df['Pass'] = df['Pass'].astype(str).str.lower().map({'true': True, 'false': False})
    return df


def make_key(row):
    detail = '' if pd.isna(row['Detail']) or row['Detail'] == '' else f" ({row['Detail']})"
    return f"{row['Test']}{detail}"


def pivot_data(df):
    df = df.copy()
    df['Key'] = df.apply(make_key, axis=1)
    pivot_p = df.pivot_table(index='Key', columns='Generator', values='p_value', aggfunc='first')
    pivot_pass = df.pivot_table(index='Key', columns='Generator', values='Pass', aggfunc='first')
    summary = pivot_p.join(pivot_pass, lsuffix='_p', rsuffix='_pass')
    return summary.reset_index()


def compare_per_test(df_wide):
    # Identify generator names
    gen_cols = [c for c in df_wide.columns if not c in ('Key',)]
    # find the two generators (assume exactly two)
    gens = sorted({c.rsplit('_', 1)[0] for c in gen_cols})
    if len(gens) != 2:
        raise RuntimeError(f"Se esperaban 2 generadores, se encontraron: {gens}")
    g1, g2 = gens
    out_rows = []
    for _, row in df_wide.iterrows():
        p1 = row.get(f"{g1}")
        p2 = row.get(f"{g2}")
        pass1 = row.get(f"{g1}_pass") if f"{g1}_pass" in row else None
        pass2 = row.get(f"{g2}_pass") if f"{g2}_pass" in row else None
        winner = None
        if pd.isna(p1) and pd.isna(p2):
            winner = 'N/A'
        elif pd.isna(p1):
            winner = g2
        elif pd.isna(p2):
            winner = g1
        else:
            if p1 > p2:
                winner = g1
            elif p2 > p1:
                winner = g2
            else:
                winner = 'Tie'
        out_rows.append({
            'Test': row['Key'],
            f'p_value_{g1}': p1,
            f'p_value_{g2}': p2,
            f'pass_{g1}': pass1,
            f'pass_{g2}': pass2,
            'winner_by_pvalue': winner,
            'p_diff': (p1 - p2) if (not pd.isna(p1) and not pd.isna(p2)) else np.nan,
        })
    return pd.DataFrame(out_rows), (g1, g2)


def overall_paired_test(df_comp, gens):
    g1, g2 = gens
    a = df_comp[f'p_value_{g1}']
    b = df_comp[f'p_value_{g2}']
    mask = a.notna() & b.notna()
    a = a[mask]
    b = b[mask]
    n = len(a)
    if n < 3:
        return {'method': 'insufficient_data', 'n': n}
    diffs = a - b
    # Test normality of differences
    try:
        shapiro_p = stats.shapiro(diffs).pvalue
    except Exception:
        shapiro_p = np.nan
    # Choose test
    if (not math.isnan(shapiro_p)) and shapiro_p > 0.05 and n >= 3:
        # paired t-test
        tstat, pval = stats.ttest_rel(a, b)
        method = 'paired_ttest'
    else:
        # Wilcoxon signed-rank test (non-parametric)
        try:
            stat, pval = stats.wilcoxon(a, b)
            tstat = stat
            method = 'wilcoxon'
        except Exception:
            # fallback to permutation-like simple test: sign test via binomial
            signs = np.sign(diffs)
            npos = np.sum(signs > 0)
            pval = stats.binom_test(npos, n)
            tstat = npos
            method = 'sign_test'
    return {
        'method': method,
        'n': n,
        'statistic': float(tstat),
        'pvalue': float(pval),
        'shapiro_p': float(shapiro_p) if not math.isnan(shapiro_p) else None,
    }


def save_plots(df_comp, gens):
    g1, g2 = gens
    # Boxplot of p-values per generator
    vals = df_comp[[f'p_value_{g1}', f'p_value_{g2}']]
    dfm = vals.melt(var_name='generator', value_name='p_value')
    dfm['generator'] = dfm['generator'].str.replace('p_value_', '')
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='generator', y='p_value', data=dfm)
    sns.stripplot(x='generator', y='p_value', data=dfm, color='k', alpha=0.4)
    plt.ylim(-0.05, 1.05)
    plt.title('Distribución de p-values por generador')
    out1 = os.path.join(IMG_DIR, 'boxplot_pvalues.png')
    plt.savefig(out1, bbox_inches='tight', dpi=150)
    plt.close()

    # Scatter paired
    mask = df_comp[[f'p_value_{g1}', f'p_value_{g2}']].notna().all(axis=1)
    plt.figure(figsize=(6, 6))
    plt.scatter(df_comp.loc[mask, f'p_value_{g1}'], df_comp.loc[mask, f'p_value_{g2}'])
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel(g1)
    plt.ylabel(g2)
    plt.title('P-values pareados: x='+g1+' vs y='+g2)
    out2 = os.path.join(IMG_DIR, 'scatter_pvalues.png')
    plt.savefig(out2, bbox_inches='tight', dpi=150)
    plt.close()

    return [out1, out2]


def main():
    df = load_data(CSV_IN)
    df_wide = pivot_data(df)
    df_comp, gens = compare_per_test(df_wide)
    stats_res = overall_paired_test(df_comp, gens)
    df_comp.to_csv(OUT_CSV, index=False)
    imgs = save_plots(df_comp, gens)

    # Small textual report
    report = os.path.join(ROOT, 'analysis_report.txt')
    with open(report, 'w', encoding='utf8') as f:
        f.write('Resumen del análisis de generadores\n')
        f.write('Generadores comparados: %s vs %s\n' % (gens[0], gens[1]))
        f.write('Número de tests pareados usados: %d\n' % stats_res.get('n', 0))
        f.write('Prueba global usada: %s\n' % stats_res.get('method'))
        f.write('Estadístico: %s\n' % stats_res.get('statistic'))
        f.write('p-value (global): %s\n' % stats_res.get('pvalue'))
        f.write('\nConclusión por test (primeras 20 filas):\n')
        f.write(df_comp.head(20).to_string(index=False))
    # Print short summary
    print('Done.')
    print('Summary CSV:', OUT_CSV)
    print('Saved images:', ', '.join(imgs))
    print('Report:', report)


if __name__ == '__main__':
    main()
