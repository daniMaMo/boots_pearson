"""
comparacion.py
==============
Comparación de tipos de distribución de Pearson según tamaño de ventana.

Funciones públicas:
    compute_window_types(log_returns, window_size, step) → pd.DataFrame
    print_comparison_table(results, step)
    plot_comparison(results, step, output_path)
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

from bootstrap import pearson_type


# =============================================================================
# CONSTANTES
# =============================================================================

TYPE_ORDER  = ["I", "II", "III", "IV", "V", "VI", "VII", "normal"]
TYPE_COLORS = {
    "I"     : "#3B5BA5",
    "II"    : "#1D9E75",
    "III"   : "#D85A30",
    "IV"    : "#8E44AD",
    "V"     : "#D4A017",
    "VI"    : "#E74C3C",
    "VII"   : "#2C3E50",
    "normal": "#7F8C8D",
}


# =============================================================================
# CÁLCULO DE TIPOS POR VENTANA
# =============================================================================

def compute_window_types(
    log_returns : np.ndarray,
    window_size : int,
    step        : int,
) -> pd.DataFrame:
    """
    Calcula los momentos y tipo de Pearson para cada ventana, sin muestrear.

    Parámetros
    ----------
    log_returns : array de log-retornos diarios
    window_size : días por ventana
    step        : salto entre ventanas

    Retorna
    -------
    pd.DataFrame con columnas: window_start, window_end, mean, std,
                               skewness, kurtosis_excess, pearson_type
    """
    n       = len(log_returns)
    records = []
    for i in range(0, n - window_size + 1, step):
        w  = log_returns[i : i + window_size]
        sk = stats.skew(w)
        ku = stats.kurtosis(w)
        records.append({
            "window_start"   : i,
            "window_end"     : i + window_size - 1,
            "mean"           : np.mean(w),
            "std"            : np.std(w, ddof=1),
            "skewness"       : sk,
            "kurtosis_excess": ku,
            "pearson_type"   : pearson_type(sk, ku),
        })
    return pd.DataFrame(records)


# =============================================================================
# TABLA COMPARATIVA EN CONSOLA
# =============================================================================

def print_comparison_table(results: dict, step: int) -> None:
    """
    Imprime tabla comparativa de tipos por tamaño de ventana.

    Parámetros
    ----------
    results : dict {window_size: pd.DataFrame} generado por compute_window_types
    step    : salto utilizado (solo para mostrar en el encabezado)
    """
    windows = sorted(results.keys())

    print("\n" + "="*70)
    print(f"COMPARACIÓN DE TIPOS DE PEARSON POR TAMAÑO DE VENTANA  (step={step})")
    print("="*70)

    header = f"{'Tipo':<8}" + "".join(f"  {w:>10}d" for w in windows)
    print(header)
    print("-" * len(header))

    for t in TYPE_ORDER:
        fila = f"{('Tipo ' + t):<8}"
        hay_datos = False
        for w in windows:
            c   = results[w]["pearson_type"].value_counts().get(t, 0)
            pct = 100 * c / len(results[w]) if len(results[w]) > 0 else 0
            fila += f"  {c:>5,} {pct:>4.1f}%"
            if c > 0:
                hay_datos = True
        if hay_datos:
            print(fila)

    print("-" * len(header))
    fila_total = f"{'Total':<8}"
    for w in windows:
        fila_total += f"  {len(results[w]):>5,}      "
    print(fila_total)
    print()


# =============================================================================
# VISUALIZACIÓN COMPARATIVA
# =============================================================================

def plot_comparison(results: dict, step: int, output_path: str) -> None:
    """
    Genera figura comparativa de 3 filas para todos los tamaños de ventana.

    Parámetros
    ----------
    results     : dict {window_size: pd.DataFrame} de compute_window_types
    step        : salto utilizado (para el título)
    output_path : ruta donde se guarda la figura (.png)
    """
    windows = sorted(results.keys())
    n_w     = len(windows)

    fig = plt.figure(figsize=(5 * n_w, 13))
    fig.suptitle(
        f"Comparación de tipos de Pearson según tamaño de ventana — DJIA  (step={step})",
        fontsize=13, fontweight="bold", y=0.99,
    )
    gs = gridspec.GridSpec(3, n_w, figure=fig, hspace=0.52, wspace=0.38)

    # ── Fila 1: barras de frecuencia por tipo ────────────────────────────────
    for j, w in enumerate(windows):
        ax     = fig.add_subplot(gs[0, j])
        df     = results[w]
        counts = df["pearson_type"].value_counts()
        total  = len(df)
        tipos  = [t for t in TYPE_ORDER if t in counts.index]
        vals   = [counts[t] for t in tipos]
        cols   = [TYPE_COLORS[t] for t in tipos]
        bars   = ax.bar(tipos, vals, color=cols, edgecolor="white")
        for bar, t in zip(bars, tipos):
            pct = 100 * counts[t] / total
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + total * 0.005,
                f"{pct:.1f}%",
                ha="center", va="bottom", fontsize=7, fontweight="bold",
            )
        ax.set_title(f"Ventana = {w} días\n({total:,} ventanas)", fontsize=9)
        ax.set_xlabel("Tipo Pearson", fontsize=8)
        ax.set_ylabel("Nº ventanas", fontsize=8)
        ax.tick_params(labelsize=8)

    # ── Fila 2: evolución de kurtosis por posición ───────────────────────────
    for j, w in enumerate(windows):
        ax = fig.add_subplot(gs[1, j])
        df = results[w]
        ax.plot(df["window_start"], df["kurtosis_excess"],
                linewidth=0.7, color="#8E44AD", alpha=0.85)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.fill_between(df["window_start"], df["kurtosis_excess"],
                        where=df["kurtosis_excess"] < 0,
                        alpha=0.15, color="#3B5BA5", label="Platicúrtica (<0)")
        ax.fill_between(df["window_start"], df["kurtosis_excess"],
                        where=df["kurtosis_excess"] > 0,
                        alpha=0.15, color="#E74C3C", label="Leptocúrtica (>0)")
        ax.set_title(f"Kurtosis — ventana {w} días", fontsize=9)
        ax.set_xlabel("Posición", fontsize=8)
        ax.set_ylabel("Kurtosis excess", fontsize=8)
        ax.tick_params(labelsize=8)
        if j == 0:
            ax.legend(fontsize=6)

    # ── Fila 3 izquierda: composición apilada % ───────────────────────────────
    ax_stack = fig.add_subplot(gs[2, :n_w//2])
    bottom   = np.zeros(n_w)
    for t in TYPE_ORDER:
        pcts = [
            100 * results[w]["pearson_type"].value_counts().get(t, 0) / len(results[w])
            for w in windows
        ]
        if any(p > 0 for p in pcts):
            ax_stack.bar(
                range(n_w), pcts, bottom=bottom,
                color=TYPE_COLORS[t], label=f"Tipo {t}",
                edgecolor="white", linewidth=0.3,
            )
            bottom += np.array(pcts)
    ax_stack.set_xticks(range(n_w))
    ax_stack.set_xticklabels([f"{w}d" for w in windows], fontsize=9)
    ax_stack.set_ylabel("% de ventanas", fontsize=9)
    ax_stack.set_title("Composición de tipos\npor tamaño de ventana", fontsize=10)
    ax_stack.legend(fontsize=7, loc="upper right", ncol=2)
    ax_stack.set_ylim(0, 108)

    # ── Fila 3 derecha: distribución de kurtosis superpuesta ─────────────────
    ax_hist   = fig.add_subplot(gs[2, n_w//2:])
    palette   = ["#3B5BA5", "#1D9E75", "#D85A30", "#8E44AD",
                 "#D4A017", "#E74C3C", "#2C3E50", "#7F8C8D"]
    for w, color in zip(windows, palette):
        ax_hist.hist(
            results[w]["kurtosis_excess"], bins=60, density=True,
            alpha=0.45, color=color, label=f"{w} días",
        )
    ax_hist.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax_hist.set_title("Distribución de kurtosis\npor tamaño de ventana", fontsize=10)
    ax_hist.set_xlabel("Kurtosis excess", fontsize=9)
    ax_hist.set_ylabel("Densidad", fontsize=9)
    ax_hist.legend(fontsize=8)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Gráfica comparativa guardada en: {output_path}")