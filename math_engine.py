# math_engine.py
import streamlit as st
import numpy as np
from scipy.stats import poisson, nbinom
from scipy.optimize import minimize
from collections import Counter
import warnings
from typing import List, Dict, Tuple, Optional, Any
import os

# Suprimir avisos de PyTensor y muestreo
warnings.filterwarnings("ignore", message=".*PyTensor.*")
warnings.filterwarnings("ignore", message=".*effective sample size.*")
warnings.filterwarnings("ignore", message=".*divergencias.*")

FORCE_MLE = os.environ.get('STREAMLIT_CLOUD', False) or os.environ.get('DEEPXG_FORCE_MLE', False)

try:
    if not FORCE_MLE:
        import pymc as pm
        import arviz as az
        PYMC_AVAILABLE = True
    else:
        PYMC_AVAILABLE = False
except ImportError:
    PYMC_AVAILABLE = False
    warnings.warn("PyMC no instalado. Usando máxima verosimilitud tradicional.")


class MotorMatematico:
    def __init__(self, league_avg: float = 2.5, draw_rate_real: float = 0.25,
                 liga_id: Optional[int] = None, matches_for_estimation: Optional[List[Dict[str, Any]]] = None,
                 glicko: Optional[Any] = None):
        self.liga_id = liga_id
        self.league_avg = league_avg
        self.draw_rate_real = draw_rate_real
        self.team_strengths: Dict[str, Tuple[float, float]] = {}
        self.team_strengths_home: Dict[str, Tuple[float, float]] = {}
        self.team_strengths_away: Dict[str, Tuple[float, float]] = {}
        self.team_uncertainties: Dict[str, Tuple[float, float]] = {}
        self.home_advantage_raw = 0.2
        self.home_advantage_std = 0.05
        self.rho = 0.0
        self.alpha = 0.05
        self.pi_l = 0.06
        self.pi_v = 0.09
        self.glicko = glicko

        if matches_for_estimation and len(matches_for_estimation) > 10:
            if PYMC_AVAILABLE:
                try:
                    self._estimate_team_parameters_bayesian(matches_for_estimation)
                except Exception as e:
                    warnings.warn(f"Falló la inferencia bayesiana: {e}. Usando MLE.")
                    self._estimate_team_parameters(matches_for_estimation)
            else:
                self._estimate_team_parameters(matches_for_estimation)
        else:
            self.rho = self._optimizar_rho_simple(draw_rate_real, league_avg)
            if liga_id:
                st.session_state[f'rho_{liga_id}'] = self.rho

        if self.glicko:
            self._build_local_strengths_from_glicko()

    def _build_local_strengths_from_glicko(self):
        for team in self.glicko.ratings_home.keys():
            rating_home = self.glicko.get_rating(team, is_home=True)
            rating_away = self.glicko.get_rating(team, is_home=False)
            att_home = np.log(max(0.5, rating_home / 1000.0))
            def_home = np.log(max(0.5, rating_home / 1000.0)) * 0.8
            att_away = np.log(max(0.5, rating_away / 1000.0)) * 0.8
            def_away = np.log(max(0.5, rating_away / 1000.0))
            self.team_strengths_home[team] = (att_home, def_home)
            self.team_strengths_away[team] = (att_away, def_away)

    def _estimate_team_parameters_bayesian(self, matches: List[Dict[str, Any]]) -> None:
        teams = set()
        for m in matches:
            teams.add(m['home'])
            teams.add(m['away'])
        teams = sorted(list(teams))
        n_teams = len(teams)
        team_to_idx = {t: i for i, t in enumerate(teams)}

        home_idx = []
        away_idx = []
        home_goals = []
        away_goals = []

        for m in matches:
            home_idx.append(team_to_idx[m['home']])
            away_idx.append(team_to_idx[m['away']])
            home_goals.append(m['home_goals'])
            away_goals.append(m['away_goals'])

        home_idx = np.array(home_idx)
        away_idx = np.array(away_idx)
        home_goals = np.array(home_goals)
        away_goals = np.array(away_goals)

        with pm.Model() as model:
            # Priors no centrados para mejorar geometría del muestreo
            mu_att = pm.Normal('mu_att', mu=0, sigma=0.5)
            sigma_att = pm.HalfNormal('sigma_att', sigma=0.3)
            mu_def = pm.Normal('mu_def', mu=0, sigma=0.5)
            sigma_def = pm.HalfNormal('sigma_def', sigma=0.3)

            # Parámetros offset (no centrados)
            att_offset = pm.Normal('att_offset', mu=0, sigma=1, shape=n_teams)
            def_offset = pm.Normal('def_offset', mu=0, sigma=1, shape=n_teams)

            att = pm.Deterministic('att', mu_att + sigma_att * att_offset)
            defn = pm.Deterministic('def', mu_def + sigma_def * def_offset)
            # Restricción de media cero
            att = att - pm.math.mean(att)
            defn = defn - pm.math.mean(defn)

            home_adv = pm.Normal('home_adv', mu=0.2, sigma=0.1)
            rho = pm.Uniform('rho', lower=-0.3, upper=0.3)
            alpha = pm.Exponential('alpha', lam=10.0)

            base_rate = self.league_avg / 2

            lambda_h = pm.math.exp(att[home_idx] - defn[away_idx] + home_adv) * base_rate
            lambda_a = pm.math.exp(att[away_idx] - defn[home_idx]) * base_rate

            nb_home = pm.NegativeBinomial.dist(mu=lambda_h, alpha=alpha)
            nb_away = pm.NegativeBinomial.dist(mu=lambda_a, alpha=alpha)
            home_ll = pm.logp(nb_home, home_goals)
            away_ll = pm.logp(nb_away, away_goals)
            ll = home_ll + away_ll

            dc_factor = pm.math.switch(
                pm.math.eq(home_goals, 0) & pm.math.eq(away_goals, 0),
                1 + lambda_h * lambda_a * rho,
                pm.math.switch(
                    pm.math.eq(home_goals, 0) & pm.math.eq(away_goals, 1),
                    1 - lambda_h * rho,
                    pm.math.switch(
                        pm.math.eq(home_goals, 1) & pm.math.eq(away_goals, 0),
                        1 - lambda_a * rho,
                        pm.math.switch(
                            pm.math.eq(home_goals, 1) & pm.math.eq(away_goals, 1),
                            1 + rho,
                            1.0
                        )
                    )
                )
            )
            ll += pm.math.log(dc_factor)

            pm.Potential('likelihood', ll)

            # Muestreo robusto: 4 cadenas, target_accept alto
            trace = pm.sample(
                draws=1500,
                tune=1500,
                chains=4,        # mínimo recomendado por PyMC
                cores=1,         # evitar multiproceso en la nube
                target_accept=0.95,  # reduce divergencias
                progressbar=False,
                return_inferencedata=True
            )

        att_mean = trace.posterior['att'].mean(dim=('chain', 'draw')).values
        def_mean = trace.posterior['def'].mean(dim=('chain', 'draw')).values
        att_std = trace.posterior['att'].std(dim=('chain', 'draw')).values
        def_std = trace.posterior['def'].std(dim=('chain', 'draw')).values

        self.home_advantage_raw = float(trace.posterior['home_adv'].mean())
        self.home_advantage_std = float(trace.posterior['home_adv'].std())
        self.rho = float(trace.posterior['rho'].mean())
        self.alpha = float(trace.posterior['alpha'].mean())

        for team, idx in team_to_idx.items():
            self.team_strengths[team] = (float(att_mean[idx]), float(def_mean[idx]))
            self.team_uncertainties[team] = (float(att_std[idx]), float(def_std[idx]))

        if self.liga_id:
            st.session_state[f'rho_{self.liga_id}'] = self.rho
            st.session_state[f'home_adv_{self.liga_id}'] = self.home_advantage_raw
            st.session_state[f'bayesian_{self.liga_id}'] = True

    def _estimate_team_parameters(self, matches: List[Dict[str, Any]]) -> None:
        # ... idéntico a la versión anterior (sin cambios) ...
        teams = set()
        for m in matches:
            teams.add(m['home'])
            teams.add(m['away'])
        teams = sorted(list(teams))
        n = len(teams)
        team_to_idx = {t: i for i, t in enumerate(teams)}

        init = np.concatenate([np.zeros(n), np.zeros(n), [0.2], [0.0], [np.log(0.05)]])
        bounds = [(-3, 3)] * (2 * n) + [(0, 1.5), (-0.5, 0.5), (np.log(0.01), np.log(0.5))]
        constraints = [{'type': 'eq', 'fun': lambda x: np.mean(x[:n])}]

        def log_lik(params: np.ndarray) -> float:
            att = params[:n]
            defn = params[n:2 * n]
            home = params[2 * n]
            rho = params[2 * n + 1]
            alpha = np.exp(params[2 * n + 2])
            base = self.league_avg / 2.0

            ll = 0.0
            for m in matches:
                h = team_to_idx[m['home']]
                a = team_to_idx[m['away']]
                lambda_h = np.exp(att[h] - defn[a] + home) * base
                lambda_a = np.exp(att[a] - defn[h]) * base

                gh = m['home_goals']
                ga = m['away_goals']

                def nb_pmf(x, lam, alpha):
                    if alpha < 1e-6:
                        return poisson.pmf(x, lam)
                    r = 1.0 / alpha
                    p = 1.0 / (1.0 + alpha * lam)
                    return nbinom.pmf(x, r, p)

                p_h = nb_pmf(gh, lambda_h, alpha)
                p_a = nb_pmf(ga, lambda_a, alpha)
                p = p_h * p_a

                if gh == 0 and ga == 0:
                    p *= (1 + lambda_h * lambda_a * rho)
                elif gh == 0 and ga == 1:
                    p *= (1 - lambda_h * rho)
                elif gh == 1 and ga == 0:
                    p *= (1 - lambda_a * rho)
                elif gh == 1 and ga == 1:
                    p *= (1 + rho)

                ll += np.log(max(p, 1e-15))

            reg = 0.001 * (np.sum(att ** 2) + np.sum(defn ** 2))
            return -(ll - reg)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(log_lik, init, method='L-BFGS-B', bounds=bounds, constraints=constraints,
                           options={'maxiter': 500, 'ftol': 1e-8})

        if res.success:
            att = res.x[:n]
            defn = res.x[n:2 * n]
            self.home_advantage_raw = res.x[2 * n]
            self.rho = res.x[2 * n + 1]
            self.alpha = np.exp(res.x[2 * n + 2])
            for team, idx in team_to_idx.items():
                self.team_strengths[team] = (att[idx], defn[idx])
                self.team_uncertainties[team] = (0.3, 0.3)
        else:
            self.rho = self._optimizar_rho_simple(self.draw_rate_real, self.league_avg)
            self.home_advantage_raw = 0.2
            self.alpha = 0.05

        if self.liga_id:
            st.session_state[f'rho_{self.liga_id}'] = self.rho
            st.session_state[f'home_adv_{self.liga_id}'] = self.home_advantage_raw

    def _optimizar_rho_simple(self, empates_reales_pct: float, prom_liga: float) -> float:
        l_prom = max(0.1, prom_liga / 2)
        def loss(rho_test: np.ndarray) -> float:
            self.rho = float(rho_test[0])
            matriz = self.bivariate_poisson(l_prom, l_prom, size=5)
            p_empate = np.trace(matriz)
            return (p_empate - empates_reales_pct) ** 2
        res = minimize(loss, [0.0], bounds=[(-0.3, 0.2)])
        return float(res.x[0])

    def get_xg_for_match(self, home_team: str, away_team: str,
                         base_league_avg: Optional[float] = None,
                         use_uncertainty: bool = True) -> Tuple[float, float]:
        if base_league_avg is None:
            base_league_avg = self.league_avg / 2.0

        if home_team in self.team_strengths_home and away_team in self.team_strengths_away:
            att_h, def_h = self.team_strengths_home[home_team]
            att_a, def_a = self.team_strengths_away[away_team]
            xg_h = np.exp(att_h - def_a + self.home_advantage_raw) * base_league_avg
            xg_a = np.exp(att_a - def_h) * base_league_avg
            return max(0.1, xg_h), max(0.1, xg_a)

        if home_team in self.team_strengths and away_team in self.team_strengths:
            att_h, def_h = self.team_strengths[home_team]
            att_a, def_a = self.team_strengths[away_team]
            if use_uncertainty and home_team in self.team_uncertainties:
                att_h_std, def_h_std = self.team_uncertainties[home_team]
                att_a_std, def_a_std = self.team_uncertainties[away_team]
                shrinkage_h_att = 1.0 / (1.0 + att_h_std)
                shrinkage_h_def = 1.0 / (1.0 + def_h_std)
                shrinkage_a_att = 1.0 / (1.0 + att_a_std)
                shrinkage_a_def = 1.0 / (1.0 + def_a_std)
                att_h_shrunk = att_h * shrinkage_h_att
                def_h_shrunk = def_h * shrinkage_h_def
                att_a_shrunk = att_a * shrinkage_a_att
                def_a_shrunk = def_a * shrinkage_a_def
                xg_h = np.exp(att_h_shrunk - def_a_shrunk + self.home_advantage_raw) * base_league_avg
                xg_a = np.exp(att_a_shrunk - def_h_shrunk) * base_league_avg
            else:
                xg_h = np.exp(att_h - def_a + self.home_advantage_raw) * base_league_avg
                xg_a = np.exp(att_a - def_h) * base_league_avg
        else:
            xg_h = base_league_avg * 1.1
            xg_a = base_league_avg * 0.9
        return max(0.1, xg_h), max(0.1, xg_a)

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

    def calcular_kelly(self, prob_modelo: float, cuota: float, prob_real_bookie: float, stake_back: float = 0.20) -> float:
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

    def procesar(self, xg_l: float, xg_v: float, cuotas: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Dict[str, Any]:
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
        kellys = [self.calcular_kelly(p * 100, c, pr) for p, c, pr in zip([p1, px, p2], cuotas, prob_reales)]

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