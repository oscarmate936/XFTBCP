import requests
import logging
from constants import BASE_URL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def call_api(endpoint, params="", timeout=15):
    try:
        # Eliminar cualquier barra inicial del endpoint
        if endpoint.startswith('/'):
            endpoint = endpoint[1:]
        
        # Construir la URL correctamente (sin doble barra)
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
    # Asegurar que season tiene el formato correcto
    if not season:
        season = "2024-2025"
    
    # Construir parámetros correctamente
    params = f"?id={cup_id}&s={season}"
    data = call_api("eventsseason.php", params)
    
    if data and 'events' in data:
        logger.info(f"Se encontraron {len(data['events'])} partidos para la copa {cup_id} en la temporada {season}")
        return data['events']
    else:
        logger.warning(f"No se encontraron partidos para la copa {cup_id} en la temporada {season}")
        return []

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

# Mantener las funciones originales para compatibilidad
def get_league_data(league_id, format_type="split"):
    season = "2025-2026" if format_type == "split" else "2026"
    data = call_api("lookuptable.php", f"?l={league_id}&s={season}")
    return data.get('table', []) if data else []

def get_recent_form(team_id):
    data = call_api("eventslast.php", f"?id={team_id}")
    return data.get('results', []) if data else []