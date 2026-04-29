# data_processing.py
import math
import numpy as np
from datetime import datetime

def calc_elo(pts, max_pts, min_pts):
    if max_pts == min_pts:
        return 1.0
    return 0.85 + 0.30 * ((pts - min_pts) / (max_pts - min_pts))

def calc_fuerza_pitagorica(gf, gc, gamma=1.7):
    gf_clean = max(0.1, float(gf))
    gc_clean = max(0.1, float(gc))
    return (gf_clean ** gamma) / (gf_clean ** gamma + gc_clean ** gamma)

def get_rest_days(team_id, events, current_date):
    last_match = None
    for ev in events:
        if str(ev.get('idHomeTeam')) == str(team_id) or str(ev.get('idAwayTeam')) == str(team_id):
            try:
                match_date = datetime.strptime(ev['dateEvent'], '%Y-%m-%d').date()
                if match_date < current_date and (last_match is None or match_date > last_match):
                    last_match = match_date
            except:
                continue
    if last_match:
        return (current_date - last_match).days
    return 7