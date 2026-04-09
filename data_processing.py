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

def get_rest_days(team_id, events, current_date):
    last_match = None
    for ev in events:
        if ev.get('idHomeTeam') == team_id or ev.get('idAwayTeam') == team_id:
            try:
                match_date = datetime.strptime(ev['dateEvent'], '%Y-%m-%d').date()
                if match_date < current_date and (last_match is None or match_date > last_match):
                    last_match = match_date
            except:
                continue
    if last_match:
        return (current_date - last_match).days
    return 7

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