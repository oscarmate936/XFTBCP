# api_utils.py
import requests
import logging
from constants import BASE_URL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def call_api(endpoint, params="", timeout=15):
    try:
        if BASE_URL.endswith('/'):
            url = f"{BASE_URL}{endpoint}{params}"
        else:
            url = f"{BASE_URL}/{endpoint}{params}"
        logger.info(f"Llamando a: {url}")
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error en {endpoint}{params}: {e}")
        return None

def get_cup_matches(cup_id, season):
    """Obtiene todos los partidos de una copa en una temporada."""
    params = f"?id={cup_id}&s={season}"
    data = call_api("eventsseason.php", params)
    if data and 'events' in data:
        logger.info(f"Encontrados {len(data['events'])} partidos para copa {cup_id} temp {season}")
        return data['events']
    else:
        logger.warning(f"No hay partidos para copa {cup_id} temp {season}")
        return []

def get_team_cup_stats(team_id, cup_matches):
    gf = 0
    gc = 0
    pj = 0
    for ev in cup_matches:
        try:
            if str(ev.get('idHomeTeam')) == str(team_id):
                gf += int(ev['intHomeScore'])
                gc += int(ev['intAwayScore'])
                pj += 1
            elif str(ev.get('idAwayTeam')) == str(team_id):
                gf += int(ev['intAwayScore'])
                gc += int(ev['intHomeScore'])
                pj += 1
        except:
            continue
    return pj, gf, gc

# ========== MEJORAS: nuevas funciones ==========

def get_team_last_matches(team_id, limit=10):
    """Obtiene los últimos 'limit' partidos de un equipo (todas competiciones)."""
    data = call_api("eventslast.php", f"?id={team_id}")
    events = data.get('results', []) if data else []
    return events[:limit]

def get_team_fixture(team_id, next=5):
    """Obtiene los próximos partidos de un equipo."""
    data = call_api("eventsnext.php", f"?id={team_id}")
    events = data.get('events', []) if data else []
    return events[:next]

def get_team_statistics(team_id):
    """Obtiene estadísticas generales del equipo (victorias, derrotas, etc.)"""
    data = call_api("lookuptable.php", f"?id={team_id}")
    if data and 'table' in data and len(data['table']) > 0:
        return data['table'][0]
    return None

def get_round_matches(cup_id, season, round_name):
    """Obtiene partidos de una ronda específica (útil para fases finales)."""
    params = f"?id={cup_id}&s={season}&r={round_name}"
    data = call_api("eventsround.php", params)
    return data.get('events', []) if data else []

def get_home_away_ratio(team_id, league_id=None):
    """Calcula ratio de puntos como local vs total (si se provee league_id)."""
    if not league_id:
        return 0.5
    data = call_api("lookuptable.php", f"?l={league_id}&s=2025-2026")
    if not data or 'table' not in data:
        return 0.5
    for team in data['table']:
        if str(team.get('idTeam')) == str(team_id):
            try:
                home_pts = int(team.get('intHomePoints', 0))
                away_pts = int(team.get('intAwayPoints', 0))
                total = home_pts + away_pts + 0.001
                return home_pts / total
            except:
                return 0.5
    return 0.5

# Funciones auxiliares (para mantener compatibilidad)
def get_league_data(league_id, format_type="split"):
    season = "2025-2026" if format_type == "split" else "2026"
    data = call_api("lookuptable.php", f"?l={league_id}&s={season}")
    return data.get('table', []) if data else []

def get_recent_form(team_id):
    data = call_api("eventslast.php", f"?id={team_id}")
    return data.get('results', []) if data else []