"""
bootstrap.py
============
Bootstrap con ventana deslizante y distribución de Pearson de tipo automático.

Funciones públicas:
    load_dja(paths)                  → pd.Series de precios
    pearson_type(skewness, kurt_ex)  → str con el tipo ('I','II',...)
    pearson_rolling_bootstrap(...)   → dict con resultados
    plot_bootstrap(...)              → guarda la figura del bootstrap
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearson3, beta as beta_dist, invgamma, norm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# CARGA DE DATOS
# =============================================================================

def load_dja(paths: list[str]) -> pd.Series:
    """
    Carga los archivos CSV del DJIA, los une y devuelve serie de precios ordenada.

    Parámetros
    ----------
    paths : lista de rutas a los archivos CSV

    Retorna
    -------
    pd.Series con índice de fechas y valores de precio
    """
    frames = []
    for path in paths:
        df = pd.read_csv(path, skiprows=3, header=0,
                         names=["Date", "DJIA"],
                         quotechar='"')
        df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
        df["DJIA"] = pd.to_numeric(df["DJIA"], errors="coerce")
        frames.append(df)

    full = (pd.concat(frames)
              .drop_duplicates("Date")
              .sort_values("Date")
              .reset_index(drop=True)
              .dropna(subset=["DJIA"]))
    return full.set_index("Date")["DJIA"]


# =============================================================================
# DETERMINACIÓN DEL TIPO DE PEARSON
# =============================================================================

def pearson_type(skewness: float, kurtosis_excess: float) -> str:
    """
    Determina el tipo de distribución de Pearson usando el criterio κ (kappa).

    Clasificación según Ord, J.K. (1972). Families of Frequency Distributions.
    Griffin, London. Table 1.1.

    Donde:
        β₁ = skewness²
        β₂ = kurtosis_excess + 3   (kurtosis total)
        C  = 2β₂ - 3β₁ - 6        (segundo factor del denominador de κ)
        κ  = β₁(β₂+3)² / [4(4β₂-3β₁)(2β₂-3β₁-6)]

    Tabla de clasificación:
        Tipo I    : κ < 0
        Tipo IV   : 0 < κ < 1
        Tipo VI   : κ > 1
        Normal    : κ = β₁ = 0, β₂ = 3
        Tipo II   : κ = β₁ = 0, β₂ < 3   (caso simétrico del I)
        Tipo VII  : κ = β₁ = 0, β₂ > 3
        Tipo III  : κ → ∞,  C = 0         (denominador = 0)
        Tipo V    : κ = 1

    Parámetros
    ----------
    skewness        : sesgo de la muestra
    kurtosis_excess : kurtosis excess (normal = 0)

    Retorna
    -------
    Tipo como string: 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'normal'
    """
    b1 = skewness ** 2          # β₁
    b2 = kurtosis_excess + 3    # β₂ (kurtosis total)
    C  = 2*b2 - 3*b1 - 6       # segundo factor del denominador de κ

    # ── Casos simétricos: β₁ ≈ 0 ─────────────────────────────────────────────
    if b1 < 1e-6:
        if abs(b2 - 3) < 0.05: return "normal"   # κ = β₁ = 0, β₂ = 3
        elif b2 < 3:            return "II"        # caso simétrico del Tipo I
        else:                   return "VII"       # β₁ = 0, β₂ > 3

    # ── Tipo III: C = 0 → denominador = 0 → κ → ∞ ───────────────────────────
    if abs(C) < 1e-6:
        return "III"

    # ── Cálculo de κ ─────────────────────────────────────────────────────────

    kappa = b1 * (b2 + 3) ** 2 / (4 * (4 * b2 - 3 * b1) * C)

    # ── Clasificación por κ (Ord 1972, Table 1.1) ────────────────────────────
    if kappa < 0:                return "I"      # κ < 0
    elif abs(kappa - 1) < 0.01:  return "V"      # κ = 1  (caso especial)
    elif kappa < 1:              return "IV"     # 0 < κ < 1
    else:                        return "VI"     # κ > 1


# =============================================================================
# MUESTREO SEGÚN TIPO DE PEARSON
# =============================================================================

def _sample_pearson(pearson_t: str, mean: float, std: float,
                    skewness: float, kurtosis_excess: float,
                    n: int, rng) -> np.ndarray:
    """Muestrea n valores de la distribución de Pearson del tipo indicado."""
    seed = int(rng.integers(0, 2**31))
    try:
        if pearson_t == "normal":
            return norm.rvs(loc=mean, scale=std, size=n, random_state=seed)

        elif pearson_t in ("I", "II"):
            if std < 1e-10:
                return np.full(n, mean)
            a_hat = max(0.5, (mean**2 * (1 - mean) / std**2 - mean)) if 0 < mean < 1 else 2.0
            b_hat = max(0.5, a_hat * (1 - mean) / mean) if mean != 0 else 2.0
            lo, hi = mean - 4*std, mean + 4*std
            samples = beta_dist.rvs(a_hat, b_hat, size=n, random_state=seed)
            return lo + samples * (hi - lo)

        elif pearson_t == "III":
            return pearson3.rvs(skew=skewness, loc=mean, scale=std,
                                size=n, random_state=seed)

        elif pearson_t == "IV":
            # 0 < κ < 1: raíces complejas conjugadas → t de Student escalada
            df_t = max(2.1, 6 / (kurtosis_excess + 1e-6)) if kurtosis_excess > 0 else 10
            return mean + std * stats.t.rvs(df=df_t, size=n, random_state=seed)

        elif pearson_t == "V":
            # κ = 1: Gamma inversa
            if std < 1e-10 or mean == 0:
                return np.full(n, mean)
            alpha = (mean / std)**2 + 2
            scale = mean * (alpha - 1)
            return invgamma.rvs(alpha, scale=scale, size=n, random_state=seed)

        elif pearson_t == "VI":
            # κ > 1: Beta prima — aproximamos con Pearson3 asimétrico
            return pearson3.rvs(skew=skewness, loc=mean, scale=std,
                                size=n, random_state=seed)

        elif pearson_t == "VII":
            # β₁ = 0, β₂ > 3: simétrica con colas pesadas → t de Student
            df_t = max(2.1, 6 / (kurtosis_excess + 1e-6)) if kurtosis_excess > 0 else 10
            return mean + std * stats.t.rvs(df=df_t, size=n, random_state=seed)

        else:
            return norm.rvs(loc=mean, scale=std, size=n, random_state=seed)

    except Exception:
        return norm.rvs(loc=mean, scale=std, size=n, random_state=seed)


# =============================================================================
# BOOTSTRAP PRINCIPAL
# =============================================================================

def pearson_rolling_bootstrap(
    log_returns : np.ndarray,
    window_size : int,
    n_samples   : int,
    step        : int,
    seed        : int = 42,
) -> dict:
    """
    Bootstrap con ventana deslizante y tipo de Pearson automático.

    Parámetros
    ----------
    log_returns : array de log-retornos diarios
    window_size : días por ventana
    n_samples   : valores aleatorios a muestrear por ventana
    step        : salto entre ventanas
                  1            = máximo solapamiento
                  = window_size = ventanas completamente independientes
    seed        : semilla aleatoria

    Retorna
    -------
    dict con:
        simulated_returns  : np.ndarray — serie simulada completa
        window_stats       : pd.DataFrame — momentos y tipo por ventana
        samples_per_window : list[np.ndarray] — muestras de cada ventana
    """
    rng = np.random.default_rng(seed)
    n   = len(log_returns)

    if window_size >= n:
        raise ValueError(f"window_size ({window_size}) debe ser < len(log_returns) ({n})")
    if step < 1:
        raise ValueError(f"step debe ser >= 1, recibido: {step}")

    simulated_returns  = []
    samples_per_window = []
    window_stats       = []

    for i in range(0, n - window_size + 1, step):
        window   = log_returns[i : i + window_size]
        mean     = np.mean(window)
        std      = np.std(window, ddof=1)
        skewness = stats.skew(window)
        kurt_ex  = stats.kurtosis(window)
        ptype    = pearson_type(skewness, kurt_ex)
        samples  = _sample_pearson(ptype, mean, std, skewness, kurt_ex, n_samples, rng)

        window_stats.append({
            "window_start"   : i,
            "window_end"     : i + window_size - 1,
            "mean"           : mean,
            "std"            : std,
            "skewness"       : skewness,
            "kurtosis_excess": kurt_ex,
            "pearson_type"   : ptype,
        })
        samples_per_window.append(samples)
        simulated_returns.extend(samples)

    return {
        "simulated_returns" : np.array(simulated_returns),
        "window_stats"      : pd.DataFrame(window_stats),
        "samples_per_window": samples_per_window,
    }


# =============================================================================
# VISUALIZACIÓN DEL BOOTSTRAP
# =============================================================================

TYPE_COLORS = {
    "I":"#3B5BA5", "II":"#1D9E75", "III":"#D85A30",
    "IV":"#8E44AD", "V":"#D4A017", "VI":"#E74C3C",
    "VII":"#2C3E50", "normal":"#7F8C8D",
}


def plot_bootstrap(log_returns, dates, result, window_size, n_samples,
                   step, output_path: str) -> None:
    """Genera y guarda la figura de resultados del bootstrap."""
    fig = plt.figure(figsize=(15, 11))
    fig.suptitle(
        f"Bootstrap Pearson (tipo automático) — DJIA\n"
        f"Ventana: {window_size} días  |  Salto: {step}  |  "
        f"Muestras por ventana: {n_samples}  |  "
        f"Serie: {dates[0].year}–{dates[-1].year}",
        fontsize=13, fontweight="bold", y=0.99,
    )

    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.38)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[1, 2])
    ax5 = fig.add_subplot(gs[2, 0])
    ax6 = fig.add_subplot(gs[2, 1])
    ax7 = fig.add_subplot(gs[2, 2])

    sim = result["simulated_returns"]
    ws  = result["window_stats"]

    # 1. Serie original
    ax1.plot(dates[1:], log_returns, color="#3B5BA5", linewidth=0.6,
             alpha=0.7, label="Log-retornos DJIA")
    ax1.axhline(0, color="gray", linewidth=0.4, linestyle=":")
    ax1.set_title("Log-retornos diarios del DJIA", fontsize=10)
    ax1.set_ylabel("Log-retorno")
    ax1.legend(fontsize=8)

    # 2. Distribución comparada
    ax2.hist(log_returns, bins=80, density=True, alpha=0.55,
             color="#3B5BA5", label="Original")
    ax2.hist(sim, bins=80, density=True, alpha=0.50,
             color="#D85A30", label="Simulado")
    ax2.set_title("Distribución comparada", fontsize=10)
    ax2.set_xlabel("Log-retorno"); ax2.set_ylabel("Densidad")
    ax2.legend(fontsize=8)

    # 3. Barras de tipos con porcentaje
    type_counts = ws["pearson_type"].value_counts()
    total_w     = len(ws)
    colors_bar  = [TYPE_COLORS.get(t, "gray") for t in type_counts.index]
    bars = ax3.bar(type_counts.index, type_counts.values,
                   color=colors_bar, edgecolor="white")
    for bar, (t, c) in zip(bars, type_counts.items()):
        pct = 100 * c / total_w
        ax3.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + total_w*0.005,
                 f"{pct:.1f}%", ha="center", va="bottom",
                 fontsize=8, fontweight="bold")
    ax3.set_title("Frecuencia de tipos de Pearson\n(Ord, 1972)", fontsize=10)
    ax3.set_xlabel("Tipo"); ax3.set_ylabel("Nº de ventanas")

    # 4. Tipo de Pearson a lo largo del tiempo
    type_num = {"normal":0,"II":1,"I":2,"III":3,"IV":4,"V":5,"VI":6,"VII":7}
    y_types  = [type_num.get(t, 0) for t in ws["pearson_type"].values]
    s_colors = [TYPE_COLORS.get(t, "gray") for t in ws["pearson_type"].values]
    ax4.scatter(ws["window_start"], y_types, c=s_colors, s=2, alpha=0.5)
    ax4.set_yticks(list(type_num.values()))
    ax4.set_yticklabels(list(type_num.keys()), fontsize=7)
    ax4.set_title("Tipo de Pearson por ventana", fontsize=10)
    ax4.set_xlabel("Posición")

    # 5. Volatilidad rodante
    ax5.plot(ws["window_start"], ws["std"], color="#D85A30", linewidth=0.7)
    ax5.fill_between(ws["window_start"], ws["std"], alpha=0.15, color="#D85A30")
    ax5.set_title("Volatilidad (std) por ventana", fontsize=10)
    ax5.set_xlabel("Posición"); ax5.set_ylabel("Std")

    # 6. Skewness rodante
    ax6.plot(ws["window_start"], ws["skewness"], color="#1D9E75", linewidth=0.7)
    ax6.axhline(0, color="gray", linewidth=0.4, linestyle=":")
    ax6.fill_between(ws["window_start"], ws["skewness"],
                     alpha=0.15, color="#1D9E75")
    ax6.set_title("Skewness por ventana", fontsize=10)
    ax6.set_xlabel("Posición"); ax6.set_ylabel("Skewness")

    # 7. Kurtosis rodante
    ax7.plot(ws["window_start"], ws["kurtosis_excess"], color="#8E44AD", linewidth=0.7)
    ax7.axhline(0, color="gray", linewidth=0.4, linestyle=":")
    ax7.fill_between(ws["window_start"], ws["kurtosis_excess"],
                     alpha=0.12, color="#8E44AD")
    ax7.set_title("Kurtosis excess por ventana", fontsize=10)
    ax7.set_xlabel("Posición"); ax7.set_ylabel("Kurtosis")

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Gráfica bootstrap guardada en: {output_path}")