import math
import numpy as np
from datetime import datetime

def calc_fuerza_pitagorica(gf, gc, gamma=1.7):
    gf_clean = max(0.1, float(gf))
    gc_clean = max(0.1, float(gc))
    return (gf_clean ** gamma) / (gf_clean ** gamma + gc_clean ** gamma)

def calc_elo(pts, max_pts, min_pts):
    if max_pts == min_pts:
        return 1.0
    return 0.85 + 0.30 * ((pts - min_pts) / (max_pts - min_pts))

# ------------------------------------------------------------
# Funciones específicas para el cerebro de copas
# ------------------------------------------------------------
def calcular_stats_avanzadas_copa(goles, fechas, prom_liga_media=1.25, C=10.0, hl=30):
    """
    Calcula xG ponderado, volatilidad y momentum.
    Mismo cálculo que en train_cerebro_cups.py
    """
    if not goles:
        return prom_liga_media, 0.0, 0.0
    hoy = fechas[-1]
    pesos = np.array([np.exp(-(np.log(2)/hl) * max(0, (hoy - f).days)) for f in fechas])
    vals = np.array(goles)
    prom_pond = np.sum(vals * pesos) / np.sum(pesos)
    xg_bayes = ((prom_pond * np.sum(pesos)) + (prom_liga_media * C)) / (np.sum(pesos) + C)
    volatilidad = np.std(goles) if len(goles) > 1 else 0.0
    momentum = np.polyfit(np.arange(len(goles)), goles, 1)[0] if len(goles) >= 3 else 0.0
    return xg_bayes, volatilidad, momentum

def resolver_colley_simple(partidos_hist, equipos_dict):
    """
    Sistema de rating Colley (simple) para equipos.
    partidos_hist: lista de tuplas (idx_local, idx_visitante, resultado)
                   resultado: 0=local gana, 1=empate, 2=visitante gana
    equipos_dict: {nombre_equipo: idx}
    Retorna dict {nombre: rating}
    """
    n = len(equipos_dict)
    M = np.diag([2.0] * n)
    b = np.ones(n)
    for h, v, r in partidos_hist:
        M[h,h] += 1
        M[v,v] += 1
        M[h,v] -= 1
        M[v,h] -= 1
        if r == 0:      # local gana
            b[h] += 0.5
            b[v] -= 0.5
        elif r == 2:    # visitante gana
            b[h] -= 0.5
            b[v] += 0.5
    try:
        sol = np.linalg.solve(M, b)
        return {nombre: sol[idx] for nombre, idx in equipos_dict.items()}
    except:
        return {nombre: 0.5 for nombre in equipos_dict}