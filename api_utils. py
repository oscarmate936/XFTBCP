# api_utils.py
import requests
import logging
from constants import BASE_URL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def call_api(endpoint, params="", timeout=15):
    try:
        url = f"{BASE_URL}{endpoint}{params}"
        logger.debug(f"Llamando a: {url}")
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error en {endpoint}{params}: {e}")
        return None

# Funciones originales (por si acaso)
def get_league_data(league_id, format_type="split"):
    season = "2025-2026" if format_type == "split" else "2026"
    data = call_api("lookuptable.php", f"?l={league_id}&s={season}")
    return data.get('table', []) if data else []

def get_recent_form(team_id):
    data = call_api("eventslast.php", f"?id={team_id}")
    return data.get('results', []) if data else []

# Nuevas funciones para copas
def get_cup_matches(cup_id, season):
    """
    Obtiene todos los partidos de una copa en una temporada.
    Ejemplo: cup_id=4480 (Champions League), season="2024-2025"
    """
    data = call_api("eventsseason.php", f"?id={cup_id}&s={season}")
    return data.get('events', []) if data else []

def get_team_cup_stats(team_id, cup_matches):
    """Calcula estadísticas de un equipo SOLO en los partidos de copa proporcionados."""
    gf = 0
    gc = 0
    pj = 0
    for ev in cup_matches:
        try:
            if ev.get('idHomeTeam') == team_id:
                gf += int(ev['intHomeScore'])
                gc += int(ev['intAwayScore'])
                pj += 1
            elif ev.get('idAwayTeam') == team_id:
                gf += int(ev['intAwayScore'])
                gc += int(ev['intHomeScore'])
                pj += 1
        except:
            continue
    return pj, gf, gc