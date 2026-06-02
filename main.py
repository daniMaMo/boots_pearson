"""
main.py
=======
Punto de entrada del proyecto.

Ejecutar:
    python main.py

Parámetros configurables en la sección de abajo.
"""

import os
import numpy as np
from scipy import stats

from bootstrap   import load_dja, pearson_rolling_bootstrap, plot_bootstrap
from comparacion import compute_window_types, print_comparison_table, plot_comparison


# =============================================================================
# PARÁMETROS — modifica aquí
# =============================================================================

# Rutas de los datos
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DATA_PATHS = [
    os.path.join(DATA_DIR, "DJA.csv"),
    os.path.join(DATA_DIR, "DJA_1_.csv"),
    os.path.join(DATA_DIR, "DJA_2_.csv"),
    os.path.join(DATA_DIR, "DJA_3_.csv"),
]

# Parámetros del bootstrap
WINDOW_SIZE = 10    # días por ventana
N_SAMPLES   = 3     # valores aleatorios a muestrear por ventana
STEP        = 10    # salto entre ventanas (= WINDOW_SIZE → sin solapamiento)
RANDOM_SEED = 42    # semilla (None = completamente aleatorio)

# Tamaños de ventana para la comparación
WINDOWS_TO_COMPARE = [10, 30, 60, 120]

# Carpeta de salida para las figuras
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# MAIN
# =============================================================================

def main():

    # ── Carga de datos ────────────────────────────────────────────────────────
    print("Cargando datos del DJIA...")
    prices = load_dja(DATA_PATHS)
    print(f"  Serie completa : {prices.index[0].date()} → {prices.index[-1].date()}")
    print(f"  Observaciones  : {len(prices):,}")

    log_ret = np.diff(np.log(prices.values))
    dates   = prices.index
    print(f"  Log-retornos   : {len(log_ret):,}\n")

    # ── Bootstrap ─────────────────────────────────────────────────────────────
    print(f"Bootstrap — window_size={WINDOW_SIZE}, step={STEP}, n_samples={N_SAMPLES}")

    result = pearson_rolling_bootstrap(
        log_returns = log_ret,
        window_size = WINDOW_SIZE,
        n_samples   = N_SAMPLES,
        step        = STEP,
        seed        = RANDOM_SEED,
    )

    sim = result["simulated_returns"]
    ws  = result["window_stats"]

    print(f"  Ventanas procesadas      : {len(ws):,}")
    print(f"  Valores simulados totales: {len(sim):,}")
    print(f"  (= {len(ws):,} ventanas × {N_SAMPLES} muestras)\n")

    print("Estadísticas de la simulación:")
    print(f"  Media    : {sim.mean():.6f}")
    print(f"  Std      : {sim.std():.6f}")
    print(f"  Skewness : {stats.skew(sim):.4f}")
    print(f"  Kurtosis : {stats.kurtosis(sim):.4f}\n")

    print("Distribución de tipos de Pearson usados:")
    for t, c in ws["pearson_type"].value_counts().items():
        print(f"  Tipo {t:6s}: {c:6,} ventanas ({100*c/len(ws):.1f}%)")

    plot_bootstrap(
        log_returns = log_ret,
        dates       = dates,
        result      = result,
        window_size = WINDOW_SIZE,
        n_samples   = N_SAMPLES,
        step        = STEP,
        output_path = os.path.join(OUTPUT_DIR, "bootstrap.png"),
    )

    # ── Comparación de ventanas ───────────────────────────────────────────────
    print(f"\nComparación de ventanas: {WINDOWS_TO_COMPARE}  (step={STEP})")

    comparison = {}
    for w in WINDOWS_TO_COMPARE:
        comparison[w] = compute_window_types(log_ret, w, STEP)
        print(f"  Ventana {w:>3}d: {len(comparison[w]):,} ventanas")

    print_comparison_table(comparison, STEP)

    plot_comparison(
        results     = comparison,
        step        = STEP,
        output_path = os.path.join(OUTPUT_DIR, "comparacion_ventanas.png"),
    )


if __name__ == "__main__":
    main()
