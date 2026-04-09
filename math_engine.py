# math_engine.py
import streamlit as st
import numpy as np
from scipy.stats import poisson
from scipy.optimize import minimize
from collections import Counter

class MotorMatematico:
    def __init__(self, league_avg=2.5, draw_rate_real=0.25, liga_id=None):
        self.liga_id = liga_id
        if liga_id and f'rho_{liga_id}' in st.session_state:
            self.rho = st.session_state[f'rho_{liga_id}']
        else:
            self.rho = self.optimizar_rho(draw_rate_real, league_avg)
            if liga_id:
                st.session_state[f'rho_{liga_id}'] = self.rho
        self.pi_l = 0.06
        self.pi_v = 0.09

    def optimizar_rho(self, empates_reales_pct, prom_liga):
        l_prom = max(0.1, prom_liga / 2)
        def loss(rho_test):
            rho_val = rho_test[0]
            p_empate = 0
            for i in range(5):
                p = poisson.pmf(i, l_prom) * poisson.pmf(i, l_prom)
                if i == 0:
                    p *= max(0, 1 - (l_prom**2) * rho_val)
                elif i == 1:
                    p *= max(0, 1 - rho_val)
                p_empate += p
            return (p_empate - empates_reales_pct)**2
        res = minimize(loss, [0.0], bounds=[(-0.3, 0.2)])
        return float(res.x[0])

    def bivariate_poisson(self, l1, l2, size=12):
        l1 = max(0.01, l1)
        l2 = max(0.01, l2)
        k = np.arange(size)
        zip_l = (1 - self.pi_l) * poisson.pmf(k, l1)
        zip_l[0] += self.pi_l
        zip_v = (1 - self.pi_v) * poisson.pmf(k, l2)
        zip_v[0] += self.pi_v
        matriz = np.outer(zip_l, zip_v)
        tau = self.rho
        if l1 > 0 and l2 > 0:
            matriz[0,0] *= max(0, 1 - l1 * l2 * tau)
            if size > 1:
                matriz[0,1] *= max(0, 1 + l1 * tau)
                matriz[1,0] *= max(0, 1 + l2 * tau)
            if size > 1:
                matriz[1,1] *= max(0, 1 - tau)
        total = matriz.sum()
        if total == 0:
            return np.ones((size, size)) / (size*size)
        return np.clip(matriz / total, 0, None)

    def desvig_odds(self, cuotas):
        cuotas = [max(c, 1.01) for c in cuotas]
        inv_odds = [1/c for c in cuotas]
        margen = sum(inv_odds)
        if margen <= 0:
            return [0.33, 0.33, 0.33]
        return [p / margen for p in inv_odds]

    def calcular_kelly(self, prob_modelo, cuota, prob_real_bookie, stake_back=0.20):
        if cuota <= 1.0:
            return 0
        p = prob_modelo / 100.0
        if p <= prob_real_bookie:
            return 0
        b = cuota - 1
        kelly = (b * p - (1 - p)) / b
        return max(0, kelly * stake_back) * 100

    def simular_goles_zinb(self, xg, sims, pi_zero):
        xg = max(0.01, xg)
        alpha_disp = 0.12
        varianza = xg + alpha_disp * (xg**2)
        ceros_estructurales = np.random.rand(sims) < pi_zero
        if varianza > xg + 0.01:
            p = xg / varianza
            n = (xg**2) / (varianza - xg)
            base_sims = np.random.negative_binomial(n, p, sims)
        else:
            base_sims = np.random.poisson(xg, sims)
        return np.where(ceros_estructurales, 0, base_sims)

    def procesar(self, xg_l, xg_v, cuotas=(1.0, 1.0, 1.0)):
        size = 12
        matriz = self.bivariate_poisson(xg_l, xg_v, size)
        p1 = np.sum(np.tril(matriz, -1))
        px = np.trace(matriz)
        p2 = np.sum(np.triu(matriz, 1))
        total_prob = p1 + px + p2
        if total_prob > 0:
            p1, px, p2 = p1/total_prob, px/total_prob, p2/total_prob
        g_lines = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
        g_probs = {}
        ii, jj = np.indices(matriz.shape)
        score_sum = ii + jj
        for line in g_lines:
            over_p = matriz[score_sum > line].sum()
            under_p = 1 - over_p
            g_probs[line] = (over_p * 100, under_p * 100)
        btts = np.sum(matriz[1:, 1:])
        marcadores = [(f"{i}-{j}", matriz[i,j]*100) for i in range(5) for j in range(5)]
        sims = 100000
        sim_h = self.simular_goles_zinb(xg_l, sims, pi_zero=self.pi_l)
        sim_v = self.simular_goles_zinb(xg_v, sims, pi_zero=self.pi_v)
        tot_sim = sim_h + sim_v
        prob_reales = self.desvig_odds(cuotas)
        evs = [((p) * c) - 1 for p, c in zip([p1, px, p2], cuotas)]
        kellys = [self.calcular_kelly(p*100, c, pr) for p, c, pr in zip([p1, px, p2], cuotas, prob_reales)]
        return {
            "1X2": (p1*100, px*100, p2*100),
            "DC": ((p1+px)*100, (p2+px)*100, (p1+p2)*100),
            "BTTS": (btts*100, (1-btts)*100),
            "GOLES": g_probs,
            "TOP": sorted(marcadores, key=lambda x: x[1], reverse=True)[:3],
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