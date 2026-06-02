# Bootstrap de Pearson con Ventana Deslizante — DJIA

Bootstrap no paramétrico sobre la serie histórica del Dow Jones Industrial Average (DJIA), basado en el sistema de distribuciones de Pearson con selección automática del tipo en cada ventana.

---

## Idea principal

A diferencia de los modelos nulos clásicos (Camino Aleatorio, AR(1), GARCH), donde la forma de la distribución es fija a lo largo de toda la serie, este método ajusta una distribución de Pearson distinta en cada ventana de tiempo. El tipo de distribución (I, II, III, IV, V, VI o VII) se determina automáticamente a partir de los cuatro momentos locales de la ventana: media, varianza, skewness y kurtosis.

```
Ventana [d1 ... d10] → calcula momentos → determina tipo → muestrea N valores
         [d2 ... d11] → calcula momentos → determina tipo → muestrea N valores
                    ...
```

---

## Estructura del proyecto

```
proyecto_pearson/
│
├── main.py            # Punto de entrada — ejecutar este archivo
├── bootstrap.py       # Carga de datos, clasificación de Pearson, bootstrap y visualización
├── comparacion.py     # Comparación de tipos según tamaño de ventana
├── requirements.txt   # Dependencias
│
├── data/              # Archivos CSV del DJIA
│   ├── DJA.csv        # 1896 – 1914
│   ├── DJA_1_.csv     # 1914 – 1938
│   ├── DJA_2_.csv     # 1938 – 1962
│   └── DJA_3_.csv     # 1962 – 1986
│
└── output/            # Figuras generadas (se crea automáticamente)
    ├── bootstrap.png
    └── comparacion_ventanas.png
```

---

## Instalación

```bash
pip install -r requirements.txt
```

---

## Uso

```bash
python main.py
```

Los parámetros se configuran al inicio de `main.py`:

```python
WINDOW_SIZE        = 10    # Días por ventana
N_SAMPLES          = 3     # Valores aleatorios a muestrear por ventana
STEP               = 10    # Salto entre ventanas (= WINDOW_SIZE → sin solapamiento)
RANDOM_SEED        = 42    # Semilla aleatoria (None = completamente aleatorio)
WINDOWS_TO_COMPARE = [10, 30, 60, 120]  # Tamaños de ventana para la comparación
```

---

## Sistema de clasificación de Pearson

La clasificación de tipos sigue la **Table 1.1** de:

> Ord, J.K. (1972). *Families of Frequency Distributions*. Griffin, London.

| Tipo | Condición | Distribución equivalente |
|------|-----------|--------------------------|
| I    | κ < 0 | Beta acotada |
| II   | β₁ = 0, β₂ < 3 | Beta simétrica (caso especial del I) |
| III  | C = 0 → κ → ∞ | Gamma |
| IV   | 0 < κ < 1 | Sin forma estándar (raíces complejas) |
| V    | κ = 1 | Gamma inversa |
| VI   | κ > 1 | Beta prima |
| VII  | β₁ = 0, β₂ > 3 | t de Student generalizada |
| Normal | β₁ = 0, β₂ = 3 | Caso límite |

Donde:
- β₁ = skewness²
- β₂ = kurtosis excess + 3
- C  = 2β₂ − 3β₁ − 6
- κ  = β₁(β₂+3)² / [4(4β₂−3β₁) · C]

---

## Salidas

Al ejecutar `main.py` se generan dos figuras en la carpeta `output/`:

**`bootstrap.png`** — Resultados del bootstrap con los parámetros configurados:
- Serie original de log-retornos del DJIA
- Distribución comparada (original vs. simulada)
- Frecuencia de tipos de Pearson usados con porcentajes
- Evolución del tipo, volatilidad, skewness y kurtosis por ventana

**`comparacion_ventanas.png`** — Comparación entre los tamaños de ventana definidos en `WINDOWS_TO_COMPARE`:
- Frecuencia de tipos por tamaño de ventana
- Evolución de kurtosis por posición
- Composición apilada de tipos
- Distribución de kurtosis superpuesta

---

## Referencia

La fuente original del sistema de distribuciones de Pearson es:

> Pearson, K. (1895). *Contributions to the Mathematical Theory of Evolution, II: Skew Variation in Homogeneous Material*. Philosophical Transactions of the Royal Society of London, Series A, 186, 343–414.
> DOI: https://doi.org/10.1098/rsta.1895.0010
