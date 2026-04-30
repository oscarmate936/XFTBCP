# math_engine.py
import numpy as np
from scipy.stats import poisson, nbinom
from collections import Counter
from typing import Dict, Any, Tuple, List

class MotorMatematico:
    def __init__(self, league_avg: float = 2.5,
                 rho: float = 0.0, alpha: float = 0.05,
                 pi_l: float = 0.06, pi_v: float = 0.09):
        self.league_avg = league_avg
        self.rho = rho
        self.alpha = alpha
        self.pi_l = pi_l
        self.pi_v = pi_v

    def bivariate_poisson(self, l1: float, l2: float, size: int = 12) -> np.ndarray:
        l1 = max(0.01, l1)
        l2 = max(0.01, l2)
        k = np.arange(size)

        def nb_probs(lam):
            if self.alpha < 1e-6:
                return poisson.pmf(k, lam)
            r = 1.0 / self.alpha
            p = 1.0 / (1.0 + self.alpha * lam)
            return nbinom.pmf(k, r, p)

        zip_l = (1 - self.pi_l) * nb_probs(l1)
        zip_l[0] += self.pi_l
        zip_v = (1 - self.pi_v) * nb_probs(l2)
        zip_v[0] += self.pi_v

        matriz = np.outer(zip_l, zip_v)

        diff = abs(l1 - l2) / (l1 + l2 + 0.01)
        dynamic_rho = self.rho * np.exp(-2.0 * diff)

        tau = dynamic_rho
        if l1 > 0 and l2 > 0:
            matriz[0, 0] *= max(0, 1 - l1 * l2 * tau)
            if size > 1:
                matriz[0, 1] *= max(0, 1 + l1 * tau)
                matriz[1, 0] *= max(0, 1 + l2 * tau)
            if size > 1:
                matriz[1, 1] *= max(0, 1 - tau)
        total = matriz.sum()
        if total == 0:
            return np.ones((size, size)) / (size * size)
        return np.clip(matriz / total, 0, None)

    def desvig_odds(self, cuotas: Tuple[float, float, float]) -> List[float]:
        cuotas = [max(c, 1.01) for c in cuotas]
        inv_odds = [1 / c for c in cuotas]
        margen = sum(inv_odds)
        if margen <= 0:
            return [0.33, 0.33, 0.33]
        return [p / margen for p in inv_odds]

    def calcular_kelly(self, prob_modelo: float, cuota: float,
                       prob_real_bookie: float, stake_back: float = 0.20) -> float:
        if cuota <= 1.0:
            return 0
        p = prob_modelo / 100.0
        if p <= prob_real_bookie:
            return 0
        b = cuota - 1
        kelly = (b * p - (1 - p)) / b
        return max(0, kelly * stake_back) * 100

    def simular_goles_zinb(self, xg: float, sims: int, pi_zero: float) -> np.ndarray:
        xg = max(0.01, xg)
        ceros_estructurales = np.random.rand(sims) < pi_zero
        if self.alpha > 1e-6:
            r = 1.0 / self.alpha
            p = 1.0 / (1.0 + self.alpha * xg)
            base_sims = np.random.negative_binomial(r, p, sims)
        else:
            base_sims = np.random.poisson(xg, sims)
        return np.where(ceros_estructurales, 0, base_sims)

    def procesar(self, xg_l: float, xg_v: float,
                 cuotas: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Dict[str, Any]:
        size = 12
        matriz = self.bivariate_poisson(xg_l, xg_v, size)
        p1 = np.sum(np.tril(matriz, -1))
        px = np.trace(matriz)
        p2 = np.sum(np.triu(matriz, 1))
        total_prob = p1 + px + p2
        if total_prob > 0:
            p1, px, p2 = p1 / total_prob, px / total_prob, p2 / total_prob

        g_lines = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
        g_probs = {}
        ii, jj = np.indices(matriz.shape)
        score_sum = ii + jj
        for line in g_lines:
            over_p = matriz[score_sum > line].sum()
            under_p = 1 - over_p
            g_probs[line] = (over_p * 100, under_p * 100)

        btts = np.sum(matriz[1:, 1:])
        marcadores = [(f"{i}-{j}", matriz[i, j] * 100) for i in range(5) for j in range(5)]

        sims = 100000
        sim_h = self.simular_goles_zinb(xg_l, sims, pi_zero=self.pi_l)
        sim_v = self.simular_goles_zinb(xg_v, sims, pi_zero=self.pi_v)
        tot_sim = sim_h + sim_v

        prob_reales = self.desvig_odds(cuotas)
        evs = [((p) * c) - 1 for p, c in zip([p1, px, p2], cuotas)]
        kellys = [self.calcular_kelly(p * 100, c, pr)
                  for p, c, pr in zip([p1, px, p2], cuotas, prob_reales)]

        return {
            "1X2": (p1 * 100, px * 100, p2 * 100),
            "DC": ((p1 + px) * 100, (p2 + px) * 100, (p1 + p2) * 100),
            "BTTS": (btts * 100, (1 - btts) * 100),
            "GOLES": g_probs,
            "TOP": sorted(marcadores, key=lambda x: x[1], reverse=True)[:3],
            "TOP_TODOS": marcadores,
            "MATRIZ": matriz[:6, :6] * 100,
            "KELLY": kellys,
            "EV": evs,
            "MONTECARLO": {
                "L": (sim_h > sim_v).mean() * 100,
                "X": (sim_h == sim_v).mean() * 100,
                "V": (sim_v > sim_h).mean() * 100,
                "AVG_G": np.mean(tot_sim),
                "STD_G": np.std(tot_sim),
                "ZERO_ZERO": ((sim_h == 0) & (sim_v == 0)).mean() * 100,
                "RAW_TOTALS": tot_sim,
                "MODE_G": Counter(tot_sim).most_common(1)[0][0] if len(tot_sim) > 0 else 0,
                "RAW_H": sim_h,
                "RAW_V": sim_v,
                "CS_L": (sim_v == 0).mean() * 100,
                "CS_V": (sim_h == 0).mean() * 100,
                "GOLEADA": (np.abs(sim_h - sim_v) >= 3).mean() * 100
            }
        }