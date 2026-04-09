# data_processing.py
import math
import numpy as np
from datetime import datetime

def calc_decay(events, team_id, half_life_days=30):
    if not events:
        return None, None
    gf, gc, w_sum = 0, 0, 0
    xi = math.log(2) / half_life_days
    hoy = datetime.now().date()
    for ev in events[:15]:
        try:
            if ev.get('intHomeScore') is None:
                continue
            s_h, s_a = int(ev['intHomeScore']), int(ev['intAwayScore'])
            try:
                match_date = datetime.strptime(ev.get('dateEvent', str(hoy)), '%Y-%m-%d').date()
            except:
                match_date = hoy
            dias = (hoy - match_date).days
            w = math.exp(-xi * max(0, dias))
            if ev.get('idHomeTeam') == team_id:
                gf += s_h * w
                gc += s_a * w
            else:
                gf += s_a * w
                gc += s_h * w
            w_sum += w
        except Exception:
            continue
    if w_sum == 0:
        return None, None
    return (gf / w_sum, gc / w_sum)

def calc_elo(pts, max_pts, min_pts):
    if max_pts == min_pts:
        return 1.0
    return 0.85 + 0.30 * ((pts - min_pts) / (max_pts - min_pts))

def calc_fuerza_pitagorica(gf, gc, gamma=1.7):
    gf_clean = max(0.1, float(gf))
    gc_clean = max(0.1, float(gc))
    return (gf_clean ** gamma) / (gf_clean ** gamma + gc_clean ** gamma)

def calc_h2h_factor(team1_id, team2_id, events, half_life=180):
    if not events:
        return 1.0
    goles_favor_1 = 0
    goles_favor_2 = 0
    peso_total = 0
    hoy = datetime.now().date()
    for ev in events:
        try:
            if ev.get('idHomeTeam') == team1_id and ev.get('idAwayTeam') == team2_id:
                gf = int(ev['intHomeScore'])
                ga = int(ev['intAwayScore'])
                fecha = datetime.strptime(ev['dateEvent'], '%Y-%m-%d').date()
                dias = (hoy - fecha).days
                peso = math.exp(-dias / half_life)
                goles_favor_1 += gf * peso
                goles_favor_2 += ga * peso
                peso_total += peso
            elif ev.get('idHomeTeam') == team2_id and ev.get('idAwayTeam') == team1_id:
                gf = int(ev['intHomeScore'])
                ga = int(ev['intAwayScore'])
                fecha = datetime.strptime(ev['dateEvent'], '%Y-%m-%d').date()
                dias = (hoy - fecha).days
                peso = math.exp(-dias / half_life)
                goles_favor_2 += gf * peso
                goles_favor_1 += ga * peso
                peso_total += peso
        except:
            continue
    if peso_total == 0:
        return 1.0
    ratio = (goles_favor_1 / goles_favor_2) if goles_favor_2 > 0 else 1.5
    return max(0.7, min(1.4, ratio))

# ========== MEJORA: función get_rest_days mejorada ==========
def get_rest_days(team_id, events, current_date):
    """
    Calcula los días de descanso desde el último partido del equipo.
    events: lista de partidos (pueden incluir todas las competiciones)
    """
    last_match = None
    for ev in events:
        # Verificar si el equipo participó
        if str(ev.get('idHomeTeam')) == str(team_id) or str(ev.get('idAwayTeam')) == str(team_id):
            try:
                match_date = datetime.strptime(ev['dateEvent'], '%Y-%m-%d').date()
                if match_date < current_date and (last_match is None or match_date > last_match):
                    last_match = match_date
            except:
                continue
    if last_match:
        return (current_date - last_match).days
    return 7  # valor por defecto

# ========== MEJORA: función de ponderación exponencial para goles ==========
def weighted_goals(gf_list, gc_list, dates, half_life=30):
    """Retorna (gf_pond, gc_pond) con peso exponencial decreciente."""
    if not gf_list:
        return 0.0, 0.0
    hoy = max(dates) if dates else datetime.now().date()
    pesos = np.array([math.exp(-(math.log(2)/half_life) * max(0, (hoy - d).days)) for d in dates])
    if np.sum(pesos) == 0:
        return np.mean(gf_list), np.mean(gc_list)
    gf_pond = np.sum(np.array(gf_list) * pesos) / np.sum(pesos)
    gc_pond = np.sum(np.array(gc_list) * pesos) / np.sum(pesos)
    return gf_pond, gc_pond

# ========== MEJORA: racha de goles y clean sheets ==========
def streaks_and_form(events, team_id, n=5):
    """
    Calcula racha de victorias/derrotas, clean sheets, y promedio de goles últimos n partidos.
    Retorna dict con: win_streak, loss_streak, clean_sheets, avg_gf, avg_gc
    """
    gf_list = []
    gc_list = []
    pts_list = []
    for ev in events[:n]:
        try:
            s_h = int(ev['intHomeScore'])
            s_a = int(ev['intAwayScore'])
            if str(ev.get('idHomeTeam')) == str(team_id):
                gf_list.append(s_h)
                gc_list.append(s_a)
                pts = 3 if s_h > s_a else (1 if s_h == s_a else 0)
            elif str(ev.get('idAwayTeam')) == str(team_id):
                gf_list.append(s_a)
                gc_list.append(s_h)
                pts = 3 if s_a > s_h else (1 if s_a == s_h else 0)
            else:
                continue
            pts_list.append(pts)
        except:
            continue
    if not pts_list:
        return {'win_streak':0, 'loss_streak':0, 'clean_sheets':0, 'avg_gf':0, 'avg_gc':0}
    win_streak = 0
    loss_streak = 0
    for p in pts_list:
        if p == 3:
            win_streak += 1
            loss_streak = 0
        elif p == 0:
            loss_streak += 1
            win_streak = 0
        else:
            win_streak = 0
            loss_streak = 0
    clean_sheets = sum(1 for gc in gc_list if gc == 0)
    avg_gf = np.mean(gf_list) if gf_list else 0
    avg_gc = np.mean(gc_list) if gc_list else 0
    return {'win_streak':win_streak, 'loss_streak':loss_streak,
            'clean_sheets':clean_sheets, 'avg_gf':avg_gf, 'avg_gc':avg_gc}

# ========== MEJORA: Rating Colley (para ranking en copa) ==========
def colley_rating(teams_list, matches):
    """
    teams_list: lista de nombres únicos
    matches: lista de tuplas (home, away, result) donde result: 0=local gana, 1=empate, 2=visitante gana
    Retorna diccionario {team: rating}
    """
    n = len(teams_list)
    idx = {team: i for i, team in enumerate(teams_list)}
    M = np.diag([2.0] * n)
    b = np.ones(n)
    for h, a, r in matches:
        hi, ai = idx[h], idx[a]
        M[hi,hi] += 1
        M[ai,ai] += 1
        M[hi,ai] -= 1
        M[ai,hi] -= 1
        if r == 0:  # local gana
            b[hi] += 0.5
            b[ai] -= 0.5
        elif r == 2: # visitante gana
            b[hi] -= 0.5
            b[ai] += 0.5
    try:
        ratings = np.linalg.solve(M, b)
        return {team: ratings[i] for team, i in idx.items()}
    except:
        return {team: 0.5 for team in teams_list}

class GlickoRating:
    def __init__(self, initial_rating=1500, initial_rd=350, tau=0.5):
        self.ratings = {}
        self.rd = {}
        self.tau = tau
        self.initial_rating = initial_rating
        self.initial_rd = initial_rd

    def get_rating(self, team):
        return self.ratings.get(team, self.initial_rating)

    def get_rd(self, team):
        return self.rd.get(team, self.initial_rd)

    def update(self, team, opponent, result, opponent_rd=None):
        rating = self.get_rating(team)
        rd = self.get_rd(team)
        opp_rating = self.get_rating(opponent)
        opp_rd = opponent_rd if opponent_rd is not None else self.get_rd(opponent)

        q = math.log(10) / 400
        g_rd = 1 / math.sqrt(1 + 3 * q**2 * opp_rd**2 / math.pi**2)
        e = 1 / (1 + 10**(-g_rd * (rating - opp_rating) / 400))
        d2 = 1 / (q**2 * g_rd**2 * e * (1 - e))

        new_rd = 1 / math.sqrt(1 / rd**2 + 1 / d2)
        new_rating = rating + q / (1 / rd**2 + 1 / d2) * g_rd * (result - e)
        new_rd = min(350, math.sqrt(new_rd**2 + self.tau**2))

        self.ratings[team] = new_rating
        self.rd[team] = new_rd
        return new_rating