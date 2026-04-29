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