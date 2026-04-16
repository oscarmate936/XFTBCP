# app_cup.py
# DeepXG Cup Predictor - Versión final corregida
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import joblib
import os
from datetime import datetime
import pytz
import logging
import html

from math_engine import MotorMatematico
from api_utils import (
    call_api, get_cup_matches, get_recent_form, get_team_cup_stats,
    get_team_last_matches, get_home_away_ratio
)
from data_processing import (
    calc_decay, calc_elo, calc_fuerza_pitagorica, calc_h2h_factor,
    get_rest_days, weighted_goals, streaks_and_form
)
from visual_components import render_dual_bar, render_outcome_card, apply_custom_css
from cup_context_analyzer import CupContextAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================
def get_auto_home_away_adv(cup_matches, modo_neutral):
    """Calcula factores de ventaja local y visitante basados en datos de la copa."""
    if modo_neutral:
        return 1.0, 1.0
    default_home = 1.08
    default_away = 0.92
    if not cup_matches or len(cup_matches) < 5:
        return default_home, default_away
    total_home_goles = 0
    total_away_goles = 0
    partidos = 0
    for m in cup_matches:
        try:
            home = int(m.get('intHomeScore', 0))
            away = int(m.get('intAwayScore', 0))
            total_home_goles += home
            total_away_goles += away
            partidos += 1
        except:
            continue
    if partidos == 0:
        return default_home, default_away
    avg_home = total_home_goles / partidos
    avg_away = total_away_goles / partidos
    if avg_away == 0:
        avg_away = 0.001
    raw_factor = avg_home / avg_away
    home_adv = max(0.85, min(1.25, raw_factor))
    away_adv = max(0.75, min(1.0, 1.0 / home_adv))
    return home_adv, away_adv

def convertir_hora_elsalvador(fecha_str, hora_str=None):
    tz_sv = pytz.timezone('America/El_Salvador')
    if not fecha_str:
        return None
    try:
        if 'T' in fecha_str:
            fecha_str_clean = fecha_str.replace('Z', '+00:00')
            dt_utc = datetime.fromisoformat(fecha_str_clean)
            if dt_utc.tzinfo is None:
                dt_utc = pytz.UTC.localize(dt_utc)
            dt_local = dt_utc.astimezone(tz_sv)
            return dt_local
        else:
            dt_naive = datetime.strptime(fecha_str, '%Y-%m-%d')
            if hora_str and hora_str.strip():
                partes = hora_str.split(':')
                hh = int(partes[0])
                mm = int(partes[1])
                ss = int(partes[2]) if len(partes) > 2 else 0
                dt_naive = dt_naive.replace(hour=hh, minute=mm, second=ss)
            dt_utc = pytz.UTC.localize(dt_naive)
            dt_local = dt_utc.astimezone(tz_sv)
            return dt_local
    except Exception as e:
        logger.error(f"Error convirtiendo fecha {fecha_str}: {e}")
        return None

def formatear_dia_local(dt):
    if dt is None:
        return "Fecha no disponible"
    dias = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    meses = ['enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio',
             'julio', 'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre']
    return f"{dias[dt.weekday()]} {dt.day} de {meses[dt.month-1]}"

def formatear_hora_local(dt):
    return dt.strftime("%H:%M") if dt else "??:??"

def calcular_caracteristicas_ensemble(tl, tv, prom_media_liga, gf_rec_l, gc_rec_l, gf_rec_v, gc_rec_v,
                                      elo_l, elo_v, pit_l, pit_v, home_adv, away_adv,
                                      cuota_local, cuota_empate, cuota_visita,
                                      gd_h_3, gd_a_3, gd_h_5, gd_a_5, gd_h_10, gd_a_10,
                                      avg_gf_h_3, avg_gc_h_3, avg_gf_a_3, avg_gc_a_3,
                                      btts_h_5, btts_a_5,
                                      win_streak_h, loss_streak_h, win_streak_a, loss_streak_a,
                                      racha_esp_local, racha_esp_visita,
                                      volatilidad_local, volatilidad_visita,
                                      momentum_local, momentum_visita):
    p_l = max(1, int(tl.get('intPlayed', 1)))
    p_v = max(1, int(tv.get('intPlayed', 1)))
    prom_media = max(0.1, prom_media_liga)

    alpha_l = (int(tl.get('intGoalsFor', 0)) / p_l) / prom_media if prom_media > 0 else 1.0
    alpha_v = (int(tv.get('intGoalsFor', 0)) / p_v) / prom_media if prom_media > 0 else 1.0
    beta_l = (int(tl.get('intGoalsAgainst', 0)) / p_l) / prom_media if prom_media > 0 else 1.0
    beta_v = (int(tv.get('intGoalsAgainst', 0)) / p_v) / prom_media if prom_media > 0 else 1.0

    features = {
        'ventaja_local': alpha_l * prom_media * home_adv,
        'ventaja_visita': alpha_v * prom_media * away_adv,
        'defensa_local': beta_l,
        'defensa_visita': beta_v,
        'elo_diff': elo_l - elo_v,
        'racha_esp_local': racha_esp_local,
        'racha_esp_visita': racha_esp_visita,
        'volatilidad_local': volatilidad_local,
        'volatilidad_visita': volatilidad_visita,
        'momentum_local': momentum_local,
        'momentum_visita': momentum_visita,
        'gd_h_3': gd_h_3, 'gd_a_3': gd_a_3,
        'gd_h_5': gd_h_5, 'gd_a_5': gd_a_5,
        'gd_h_10': gd_h_10, 'gd_a_10': gd_a_10,
        'avg_gf_h_3': avg_gf_h_3, 'avg_gc_h_3': avg_gc_h_3,
        'avg_gf_a_3': avg_gf_a_3, 'avg_gc_a_3': avg_gc_a_3,
        'btts_h_5': btts_h_5, 'btts_a_5': btts_a_5,
        'win_streak_h': win_streak_h, 'loss_streak_h': loss_streak_h,
        'win_streak_a': win_streak_a, 'loss_streak_a': loss_streak_a,
    }
    try:
        inv_odds = [1 / max(c, 1.01) for c in [cuota_local, cuota_empate, cuota_visita]]
        prob_impl = [p / sum(inv_odds) for p in inv_odds]
        features['prob_impl_local'] = prob_impl[0]
        features['prob_impl_empate'] = prob_impl[1]
        features['prob_impl_visita'] = prob_impl[2]
    except:
        features['prob_impl_local'] = 0.33
        features['prob_impl_empate'] = 0.33
        features['prob_impl_visita'] = 0.34
    return features

def obtener_estadisticas_avanzadas(team_id, events, max_partidos=10):
    if not events:
        return (0, 0, 0, 0.0, 0.0, 0.0, 0, 0, 0.0, 0.0, 0.0)
    gf_list = []
    gc_list = []
    pts_list = []
    for ev in events[:max_partidos]:
        try:
            if ev.get('intHomeScore') is None:
                continue
            s_h = int(ev['intHomeScore'])
            s_a = int(ev['intAwayScore'])
            if str(ev.get('idHomeTeam')) == str(team_id):
                gf_list.append(s_h)
                gc_list.append(s_a)
                pts_list.append(3 if s_h > s_a else (1 if s_h == s_a else 0))
            elif str(ev.get('idAwayTeam')) == str(team_id):
                gf_list.append(s_a)
                gc_list.append(s_h)
                pts_list.append(3 if s_a > s_h else (1 if s_a == s_h else 0))
        except:
            continue
    if not gf_list:
        return (0, 0, 0, 0.0, 0.0, 0.0, 0, 0, 0.0, 0.0, 0.0)
    n = len(gf_list)
    gd_3 = sum(gf_list[:3]) - sum(gc_list[:3]) if n >= 3 else 0
    gd_5 = sum(gf_list[:5]) - sum(gc_list[:5]) if n >= 5 else 0
    gd_10 = sum(gf_list) - sum(gc_list)
    avg_gf_3 = np.mean(gf_list[:3]) if n >= 3 else 0.0
    avg_gc_3 = np.mean(gc_list[:3]) if n >= 3 else 0.0
    btts_cnt = 0
    for i in range(min(5, n)):
        if gf_list[i] > 0 and gc_list[i] > 0:
            btts_cnt += 1
    btts_ratio = btts_cnt / min(5, n) if n > 0 else 0.0
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
    pesos = [0.1, 0.15, 0.2, 0.25, 0.3]
    racha = sum(p * w for p, w in zip(pts_list[:5], pesos[:len(pts_list[:5])])) if n >= 5 else np.mean(pts_list)
    total_goles_partido = [gf_list[i] + gc_list[i] for i in range(n)]
    vol = np.std(total_goles_partido) if n > 1 else 0.5
    if n >= 3:
        mom = np.polyfit(range(n), gf_list, 1)[0]
    else:
        mom = 0.0
    return (gd_3, gd_5, gd_10, avg_gf_3, avg_gc_3, btts_ratio,
            win_streak, loss_streak, racha, vol, mom)

def find_first_leg_match(team_home, team_away, current_event_date, cup_matches):
    """
    Busca un partido previo entre los mismos equipos en la misma copa/temporada.
    Retorna (goles_local_ida, goles_visitante_ida) o None si no se encuentra.
    """
    if not cup_matches:
        return None
    try:
        current_date = datetime.strptime(current_event_date, '%Y-%m-%d').date()
    except:
        return None

    best_match = None
    for m in cup_matches:
        try:
            m_date = datetime.strptime(m['dateEvent'], '%Y-%m-%d').date()
        except:
            continue
        if m_date >= current_date:
            continue
        if (m['strHomeTeam'] == team_home and m['strAwayTeam'] == team_away) or \
           (m['strHomeTeam'] == team_away and m['strAwayTeam'] == team_home):
            if best_match is None or m_date > best_match[0]:
                best_match = (m_date, m)
    if best_match:
        match = best_match[1]
        if match['strHomeTeam'] == team_home:
            return int(match['intHomeScore']), int(match['intAwayScore'])
        else:
            return int(match['intAwayScore']), int(match['intHomeScore'])
    return None

def generar_sugerencias(res, cuotas, nombres_equipos):
    """
    Analiza los resultados del modelo y genera una lista de sugerencias
    priorizadas por probabilidad, valor esperado y Kelly.
    """
    sugerencias = []
    prob_local, prob_empate, prob_visita = res['1X2']
    ev_local, ev_empate, ev_visita = res['EV']
    kelly_local, kelly_empate, kelly_visita = res['KELLY']
    o1, ox, o2 = cuotas

    # 1X2 básico
    outcomes = [
        ('Local', prob_local, ev_local, kelly_local, o1, '🏠'),
        ('Empate', prob_empate, ev_empate, kelly_empate, ox, '🤝'),
        ('Visitante', prob_visita, ev_visita, kelly_visita, o2, '✈️')
    ]
    for nombre, prob, ev, kelly, cuota, icono in outcomes:
        if prob > 45 or ev > 0.05 or kelly > 0.5:
            razon = []
            if prob > 45:
                razon.append(f"Alta probabilidad ({prob:.1f}%)")
            if ev > 0.05:
                razon.append(f"Valor esperado positivo ({ev:+.2f})")
            if kelly > 0.5:
                razon.append(f"Kelly {kelly:.1f}%")
            sugerencias.append({
                'mercado': '1X2',
                'seleccion': f"{icono} {nombre}",
                'prob': prob,
                'cuota': cuota,
                'ev': ev,
                'kelly': kelly,
                'razon': ' · '.join(razon) if razon else 'Recomendación del modelo'
            })

    # Doble oportunidad
    dc_local_empate, dc_visita_empate, dc_local_visita = res['DC']
    cuotas_dc = [
        (dc_local_empate, '1X', 'Local o Empate', '🏠🤝'),
        (dc_visita_empate, 'X2', 'Visitante o Empate', '✈️🤝'),
        (dc_local_visita, '12', 'Local o Visitante', '🏠✈️')
    ]
    for prob, mercado, nombre, icono in cuotas_dc:
        if prob > 65:
            sugerencias.append({
                'mercado': 'Doble Oportunidad',
                'seleccion': f"{icono} {nombre}",
                'prob': prob,
                'cuota': None,
                'ev': None,
                'kelly': None,
                'razon': f'Muy alta probabilidad ({prob:.1f}%)'
            })

    # Goles Over/Under
    for linea in [1.5, 2.5, 3.5]:
        over, under = res['GOLES'][linea]
        if over > 60:
            sugerencias.append({
                'mercado': f'Total Goles',
                'seleccion': f"🔥 Over {linea}",
                'prob': over,
                'cuota': None,
                'ev': None,
                'kelly': None,
                'razon': f'Alta probabilidad de más de {linea} goles ({over:.1f}%)'
            })
        if under > 60:
            sugerencias.append({
                'mercado': f'Total Goles',
                'seleccion': f"❄️ Under {linea}",
                'prob': under,
                'cuota': None,
                'ev': None,
                'kelly': None,
                'razon': f'Alta probabilidad de menos de {linea} goles ({under:.1f}%)'
            })

    # Ambos anotan
    btts_si, btts_no = res['BTTS']
    if btts_si > 55:
        sugerencias.append({
            'mercado': 'Ambos Anotan',
            'seleccion': '✅ Sí',
            'prob': btts_si,
            'cuota': None,
            'ev': None,
            'kelly': None,
            'razon': f'Alta probabilidad de BTTS ({btts_si:.1f}%)'
        })
    if btts_no > 55:
        sugerencias.append({
            'mercado': 'Ambos Anotan',
            'seleccion': '❌ No',
            'prob': btts_no,
            'cuota': None,
            'ev': None,
            'kelly': None,
            'razon': f'Alta probabilidad de que NO anoten ambos ({btts_no:.1f}%)'
        })

    # Hándicap asiático desde simulación
    mc = res['MONTECARLO']
    margin = mc['RAW_H'] - mc['RAW_V']
    prob_local_ah = (margin >= 1).mean() * 100
    if prob_local_ah > 55:
        sugerencias.append({
            'mercado': 'Hándicap Asiático',
            'seleccion': f"{nombres_equipos[0]} -0.5",
            'prob': prob_local_ah,
            'cuota': None,
            'ev': None,
            'kelly': None,
            'razon': f'Cubre el hándicap en {prob_local_ah:.1f}% de simulaciones'
        })
    prob_visitante_ah = (margin <= 0).mean() * 100
    if prob_visitante_ah > 55:
        sugerencias.append({
            'mercado': 'Hándicap Asiático',
            'seleccion': f"{nombres_equipos[1]} +0.5",
            'prob': prob_visitante_ah,
            'cuota': None,
            'ev': None,
            'kelly': None,
            'razon': f'No pierde en {prob_visitante_ah:.1f}% de simulaciones'
        })

    sugerencias.sort(key=lambda x: (x.get('ev') or -999, x['prob']), reverse=True)
    return sugerencias[:5]

def mostrar_panel_sugerencias(sugerencias):
    """Renderiza un panel atractivo con las sugerencias destacadas."""
    if not sugerencias:
        return

    # CSS como string separado para evitar conflictos con f-string
    st.markdown("""
    <style>
    .suggestions-panel {
        background: linear-gradient(145deg, #1E293B 0%, #0F172A 100%);
        border-radius: 24px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(59,130,246,0.3);
        box-shadow: 0 8px 20px rgba(0,0,0,0.5);
    }
    .suggestions-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #F8FAFC;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 1rem;
    }
    .suggestion-badge {
        display: inline-block;
        background: #3B82F6;
        color: white;
        border-radius: 40px;
        padding: 2px 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 10px;
    }
    .suggestion-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
    }
    .suggestion-card {
        background: #1E293B;
        border-radius: 18px;
        padding: 0.9rem 1.2rem;
        flex: 1 1 180px;
        border-left: 4px solid #3B82F6;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transition: transform 0.1s ease;
    }
    .suggestion-card:hover {
        transform: translateY(-2px);
        background: #2D3A4E;
    }
    .suggestion-mercado {
        font-size: 0.7rem;
        font-weight: 600;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .suggestion-seleccion {
        font-size: 1.25rem;
        font-weight: 800;
        color: #F8FAFC;
        margin: 6px 0 4px;
    }
    .suggestion-prob {
        font-size: 1.4rem;
        font-weight: 800;
        color: #3B82F6;
    }
    .suggestion-razon {
        font-size: 0.65rem;
        color: #94A3B8;
        margin-top: 6px;
        border-top: 1px solid rgba(255,255,255,0.1);
        padding-top: 6px;
    }
    .suggestion-ev {
        font-size: 0.75rem;
        font-weight: 700;
        color: #10B981;
    }
    </style>
    """, unsafe_allow_html=True)

    # Construir el HTML con cuidado, escapando valores dinámicos
    html_parts = []
    html_parts.append('<div class="suggestions-panel">')
    html_parts.append('<div class="suggestions-title">🎯 Sugerencias Destacadas <span class="suggestion-badge">ALTA CONFIANZA</span></div>')
    html_parts.append('<div class="suggestion-grid">')

    for sug in sugerencias:
        mercado = html.escape(sug['mercado'])
        seleccion = html.escape(sug['seleccion'])
        razon = html.escape(sug['razon'])
        prob = f"{sug['prob']:.1f}"
        ev_str = f" · EV {sug['ev']:+.2f}" if sug.get('ev') is not None else ''
        kelly_str = f" · K {sug['kelly']:.1f}%" if sug.get('kelly') is not None and sug['kelly'] > 0 else ''
        cuota_str = f" · Cuota {sug['cuota']:.2f}" if sug.get('cuota') is not None else ''
        metricas = f"{ev_str}{kelly_str}{cuota_str}"

        card = f'''
        <div class="suggestion-card">
            <div class="suggestion-mercado">{mercado}</div>
            <div class="suggestion-seleccion">{seleccion}</div>
            <div style="display: flex; align-items: baseline; gap: 8px;">
                <span class="suggestion-prob">{prob}%</span>
                <span class="suggestion-ev">{metricas}</span>
            </div>
            <div class="suggestion-razon">{razon}</div>
        </div>
        '''
        html_parts.append(card)

    html_parts.append('</div>')
    html_parts.append('</div>')

    full_html = ''.join(html_parts)
    st.markdown(full_html, unsafe_allow_html=True)

# ============================================================
# INICIALIZACIÓN DEL ESTADO DE SESIÓN
# ============================================================
for key, default in [
    ('p_copa_auto', 2.5), ('nl_auto', "Equipo Local"), ('nv_auto', "Equipo Visitante"),
    ('cup_matches_cached', []), ('draw_rate_auto', 0.25), ('tl_stats', None), ('tv_stats', None),
    ('lgf_auto', 1.2), ('lgc_auto', 1.0), ('vgf_auto', 1.1), ('vgc_auto', 1.3),
    ('elo_l', 1.0), ('elo_v', 1.0), ('pit_l', 0.5), ('pit_v', 0.5),
    ('prom_media_copa', 1.25), ('h_adv_l', 1.10), ('v_adv_v', 0.90),
    ('gd_h_3', 0), ('gd_a_3', 0), ('gd_h_5', 0), ('gd_a_5', 0), ('gd_h_10', 0), ('gd_a_10', 0),
    ('avg_gf_h_3', 0.0), ('avg_gc_h_3', 0.0), ('avg_gf_a_3', 0.0), ('avg_gc_a_3', 0.0),
    ('btts_h_5', 0.0), ('btts_a_5', 0.0),
    ('win_streak_h', 0), ('loss_streak_h', 0), ('win_streak_a', 0), ('loss_streak_a', 0),
    ('racha_esp_local', 0.0), ('racha_esp_visita', 0.0),
    ('volatilidad_local', 0.0), ('volatilidad_visita', 0.0),
    ('momentum_local', 0.0), ('momentum_visita', 0.0),
    ('rest_days_local', 7), ('rest_days_visitor', 7),
    ('cerebro_ensemble', None), ('cerebro_path', ''),
    ('first_leg_auto_local', 0), ('first_leg_auto_visitor', 0), ('first_leg_detected', False)
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ============================================================
# CARGA DEL CEREBRO (ENSEMBLE)
# ============================================================
cerebro_path = None
for path in ['quantum_cerebro_cup_final.pkl', 'quantum_cerebro_final.pkl', 'quantum_cerebro_ensemble_light.pkl']:
    if os.path.exists(path):
        cerebro_path = path
        break
if cerebro_path:
    try:
        cerebro_ensemble = joblib.load(cerebro_path)
        st.session_state['cerebro_ensemble'] = cerebro_ensemble
        st.session_state['cerebro_path'] = cerebro_path
        logger.info(f"Cerebro cargado desde {cerebro_path}")
    except Exception as e:
        st.session_state['cerebro_ensemble'] = None
        st.warning(f"Error cargando cerebro ({cerebro_path}): {e}")
else:
    st.session_state['cerebro_ensemble'] = None

apply_custom_css()
st.set_page_config(page_title="DeepXG Cup Predictor", layout="wide", initial_sidebar_state="expanded")

# ============================================================
# SIDEBAR (SELECCIÓN DE COPA Y SINCRONIZACIÓN)
# ============================================================
with st.sidebar:
    st.markdown("<h3 style='color:var(--primary); font-weight:700; margin-bottom:0;'>🏆 Data Hub - Copas</h3>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.7rem; color:var(--text-sub); margin-bottom:1rem;'>Sincroniza torneos de eliminación directa</p>", unsafe_allow_html=True)

    copas = {
        "🏆 Champions League": {"id": 4480, "season_default": "2025-2026"},
        "🏆 Europa League": {"id": 4481, "season_default": "2025-2026"},
        "🏆 Conference League": {"id": 5071, "season_default": "2025-2026"},
        "🇪🇸 Copa del Rey": {"id": 4483, "season_default": "2025-2026"},
        "🇬🇧 FA Cup": {"id": 4490, "season_default": "2025-2026"},
        "🇮🇹 Coppa Italia": {"id": 4485, "season_default": "2025-2026"},
        "🇩🇪 DFB-Pokal": {"id": 4486, "season_default": "2025-2026"},
        "🇫🇷 Coupe de France": {"id": 4487, "season_default": "2025-2026"},
        "🌍 Mundial de Clubes": {"id": 4488, "season_default": "2026"},
    }

    copa_sel = st.selectbox("Elegir Copa", list(copas.keys()))
    cup_info = copas[copa_sel]
    cup_id = cup_info["id"]
    season = st.text_input("Temporada (ej. 2025-2026 o 2025)", value=cup_info["season_default"])

    if st.button("🔄 Sincronizar Datos", use_container_width=True):
        with st.spinner(f"Obteniendo partidos de {copa_sel} temporada {season}..."):
            matches = get_cup_matches(cup_id, season)
        if matches:
            st.session_state['cup_matches_cached'] = matches
            total_goles = 0
            total_partidos = 0
            for m in matches:
                try:
                    total_goles += int(m['intHomeScore']) + int(m['intAwayScore'])
                    total_partidos += 1
                except:
                    continue
            prom_goles = total_goles / max(1, total_partidos)
            st.session_state['p_copa_auto'] = round(prom_goles, 2)
            st.session_state['draw_rate_auto'] = 0.25
            st.success(f"✅ {len(matches)} partidos sincronizados. Promedio goles: {prom_goles:.2f}")
        else:
            st.error(f"❌ No se encontraron partidos para {copa_sel} en la temporada {season}. Prueba otra temporada (ej. 2024-2025 o 2025).")
        st.rerun()

    if st.session_state.get('cup_matches_cached'):
        st.markdown("---")
        st.markdown("### 📅 Partidos de los próximos 2 días")
        all_matches = st.session_state['cup_matches_cached']
        if all_matches:
            tz_sv = pytz.timezone('America/El_Salvador')
            hoy_local = datetime.now(tz_sv).date()
            limite = hoy_local + pd.Timedelta(days=2)
            partidos_proximos = []
            for ev in all_matches:
                dt_local = convertir_hora_elsalvador(ev.get('dateEvent', ''), ev.get('strTime', ''))
                if dt_local is not None:
                    fecha_local = dt_local.date()
                    if hoy_local <= fecha_local <= limite:
                        partidos_proximos.append(ev)
                else:
                    try:
                        fecha_str = ev.get('dateEvent', '')
                        if fecha_str:
                            fecha_naive = datetime.strptime(fecha_str, '%Y-%m-%d').date()
                            if hoy_local <= fecha_naive <= limite:
                                partidos_proximos.append(ev)
                    except:
                        pass
            if partidos_proximos:
                partidos_proximos.sort(key=lambda x: (x.get('dateEvent', ''), x.get('strTime', '')))
                opciones = []
                mapeo = {}
                for ev in partidos_proximos:
                    dt_local = convertir_hora_elsalvador(ev.get('dateEvent', ''), ev.get('strTime', ''))
                    if dt_local:
                        fecha_str = formatear_dia_local(dt_local)
                        hora_str = formatear_hora_local(dt_local)
                        texto = f"{fecha_str} {hora_str} - {ev['strHomeTeam']} vs {ev['strAwayTeam']}"
                    else:
                        texto = f"{ev['dateEvent']} - {ev['strHomeTeam']} vs {ev['strAwayTeam']}"
                    opciones.append(texto)
                    mapeo[texto] = ev
                sel_match = st.selectbox("Selecciona un partido", opciones)
                evento = mapeo[sel_match]
                if st.button("📊 Cargar Estadísticas", use_container_width=True):
                    m = evento
                    st.session_state['nl_auto'], st.session_state['nv_auto'] = m['strHomeTeam'], m['strAwayTeam']
                    pj_l, gf_l, gc_l = get_team_cup_stats(m['idHomeTeam'], st.session_state['cup_matches_cached'])
                    pj_v, gf_v, gc_v = get_team_cup_stats(m['idAwayTeam'], st.session_state['cup_matches_cached'])
                    prom_media = st.session_state['p_copa_auto'] / 2
                    tl_stats = {'intPlayed': max(pj_l, 1), 'intGoalsFor': gf_l, 'intGoalsAgainst': gc_l}
                    tv_stats = {'intPlayed': max(pj_v, 1), 'intGoalsFor': gf_v, 'intGoalsAgainst': gc_v}
                    st.session_state['tl_stats'] = tl_stats
                    st.session_state['tv_stats'] = tv_stats
                    max_pts = max(pj_l*3, pj_v*3)
                    elo_l = calc_elo(gf_l - gc_l, max_pts, 0)
                    elo_v = calc_elo(gf_v - gc_v, max_pts, 0)
                    pit_l = calc_fuerza_pitagorica(gf_l, gc_l)
                    pit_v = calc_fuerza_pitagorica(gf_v, gc_v)
                    home_adv, away_adv = get_auto_home_away_adv(st.session_state['cup_matches_cached'], False)
                    xg_l_base = (gf_l / max(pj_l,1)) * elo_l * pit_l
                    xg_v_base = (gf_v / max(pj_v,1)) * elo_v * pit_v
                    st.session_state['lgf_auto'] = xg_l_base * home_adv
                    st.session_state['vgf_auto'] = xg_v_base * away_adv
                    st.session_state['lgc_auto'] = (gc_l / max(pj_l,1)) / max(elo_v, 0.1)
                    st.session_state['vgc_auto'] = (gc_v / max(pj_v,1)) / max(elo_l, 0.1)
                    st.session_state['elo_l'] = elo_l
                    st.session_state['elo_v'] = elo_v
                    st.session_state['pit_l'] = pit_l
                    st.session_state['pit_v'] = pit_v
                    st.session_state['prom_media_copa'] = prom_media
                    recent_l = get_team_last_matches(m['idHomeTeam'], limit=10)
                    recent_v = get_team_last_matches(m['idAwayTeam'], limit=10)
                    (gd_h_3, gd_h_5, gd_h_10, avg_gf_h_3, avg_gc_h_3, btts_h_5,
                     win_streak_h, loss_streak_h, racha_h, vol_h, mom_h) = obtener_estadisticas_avanzadas(m['idHomeTeam'], recent_l)
                    (gd_a_3, gd_a_5, gd_a_10, avg_gf_a_3, avg_gc_a_3, btts_a_5,
                     win_streak_a, loss_streak_a, racha_a, vol_a, mom_a) = obtener_estadisticas_avanzadas(m['idAwayTeam'], recent_v)
                    st.session_state['gd_h_3'] = gd_h_3
                    st.session_state['gd_a_3'] = gd_a_3
                    st.session_state['gd_h_5'] = gd_h_5
                    st.session_state['gd_a_5'] = gd_a_5
                    st.session_state['gd_h_10'] = gd_h_10
                    st.session_state['gd_a_10'] = gd_a_10
                    st.session_state['avg_gf_h_3'] = avg_gf_h_3
                    st.session_state['avg_gc_h_3'] = avg_gc_h_3
                    st.session_state['avg_gf_a_3'] = avg_gf_a_3
                    st.session_state['avg_gc_a_3'] = avg_gc_a_3
                    st.session_state['btts_h_5'] = btts_h_5
                    st.session_state['btts_a_5'] = btts_a_5
                    st.session_state['win_streak_h'] = win_streak_h
                    st.session_state['loss_streak_h'] = loss_streak_h
                    st.session_state['win_streak_a'] = win_streak_a
                    st.session_state['loss_streak_a'] = loss_streak_a
                    st.session_state['racha_esp_local'] = racha_h
                    st.session_state['racha_esp_visita'] = racha_a
                    st.session_state['volatilidad_local'] = vol_h
                    st.session_state['volatilidad_visita'] = vol_a
                    st.session_state['momentum_local'] = mom_h
                    st.session_state['momentum_visita'] = mom_a
                    fecha_partido = datetime.strptime(m['dateEvent'], '%Y-%m-%d').date()
                    all_events_local = get_team_last_matches(m['idHomeTeam'], limit=10)
                    all_events_visitor = get_team_last_matches(m['idAwayTeam'], limit=10)
                    rest_local = get_rest_days(m['idHomeTeam'], all_events_local, fecha_partido)
                    rest_visitor = get_rest_days(m['idAwayTeam'], all_events_visitor, fecha_partido)
                    st.session_state['rest_days_local'] = rest_local
                    st.session_state['rest_days_visitor'] = rest_visitor

                    # --- Detectar automáticamente resultado de ida ---
                    first_leg = find_first_leg_match(
                        m['strHomeTeam'], m['strAwayTeam'],
                        m['dateEvent'],
                        st.session_state['cup_matches_cached']
                    )
                    if first_leg:
                        st.session_state['first_leg_auto_local'] = first_leg[0]
                        st.session_state['first_leg_auto_visitor'] = first_leg[1]
                        st.session_state['first_leg_detected'] = True
                    else:
                        st.session_state['first_leg_detected'] = False

                    st.rerun()
            else:
                st.info("No hay partidos programados en los próximos 2 días para esta competición.")
        else:
            st.info("No hay partidos en esta competición para la temporada seleccionada.")

# ============================================================
# UI PRINCIPAL
# ============================================================
st.markdown("<h1 class='app-title'>DeepXG <span>Cup Predictor</span></h1>", unsafe_allow_html=True)
st.markdown("<p class='app-subtitle'>MOTOR ESTADÍSTICO AVANZADO PARA TORNEOS DE ELIMINACIÓN</p>", unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">⚙️ Configuración del Partido</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        p_copa = st.number_input("Goles Prom. Copa", 0.5, 5.0, float(st.session_state.get('p_copa_auto', 2.5)), 0.01)
        modo_neutral = st.toggle("🏟️ Sede Neutral", value=False)
        is_second_leg = st.checkbox("¿Partido de vuelta?")
        if is_second_leg:
            default_local = st.session_state.get('first_leg_auto_local', 0)
            default_visitante = st.session_state.get('first_leg_auto_visitor', 0)
            col_ida1, col_ida2 = st.columns(2)
            with col_ida1:
                goles_ida_local = st.number_input("Goles Local (ida)", 0, 10, default_local)
            with col_ida2:
                goles_ida_visitante = st.number_input("Goles Visitante (ida)", 0, 10, default_visitante)
            first_leg_result = (goles_ida_local, goles_ida_visitante)
            if st.session_state.get('first_leg_detected'):
                st.success(f"✅ Resultado de ida detectado automáticamente: {default_local}-{default_visitante}")
        else:
            first_leg_result = None
    with col2:
        st.markdown("#### Equipos")
        nl = st.text_input("🏠 Local", st.session_state['nl_auto'])
        nv = st.text_input("✈️ Visitante", st.session_state['nv_auto'])

    st.markdown("#### xG Esperado")
    xg_cols = st.columns(4)
    with xg_cols[0]:
        lgf = st.number_input("xG Favor L", 0.0, 5.0, st.session_state.get('lgf_auto', 1.2), step=0.05)
    with xg_cols[1]:
        lgc = st.number_input("xG Contra L", 0.0, 5.0, st.session_state.get('lgc_auto', 1.0), step=0.05)
    with xg_cols[2]:
        vgf = st.number_input("xG Favor V", 0.0, 5.0, st.session_state.get('vgf_auto', 1.1), step=0.05)
    with xg_cols[3]:
        vgc = st.number_input("xG Contra V", 0.0, 5.0, st.session_state.get('vgc_auto', 1.3), step=0.05)

    st.markdown("#### 📊 Cuotas de Mercado (1X2)")
    odd_cols = st.columns(3)
    with odd_cols[0]:
        o1 = st.number_input("Local (1)", 1.0, 50.0, 2.0, step=0.1)
    with odd_cols[1]:
        ox = st.number_input("Empate (X)", 1.0, 50.0, 3.4, step=0.1)
    with odd_cols[2]:
        o2 = st.number_input("Visita (2)", 1.0, 50.0, 3.2, step=0.1)

    ronda = st.text_input("Ronda actual (ej. Semifinal)", value="Cuartos")

    analizar_btn = st.button("🚀 PROCESAR PREDICCIÓN", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

if analizar_btn:
    res = None
    adjustments = None

    if st.session_state.get('cup_matches_cached') and st.session_state.get('tl_stats') is not None:
        try:
            analyzer = CupContextAnalyzer(
                cup_matches=st.session_state['cup_matches_cached'],
                home_team=nl,
                away_team=nv,
                current_round=ronda,
                is_second_leg=is_second_leg,
                first_leg_result=first_leg_result if is_second_leg else None
            )
            adjustments = analyzer.get_adjustments()
        except Exception as e:
            logger.error(f"Error en análisis de contexto de copa: {e}")
            adjustments = None

    home_adv_used, away_adv_used = get_auto_home_away_adv(
        st.session_state.get('cup_matches_cached', []),
        modo_neutral
    )
    st.session_state['h_adv_l'] = home_adv_used
    st.session_state['v_adv_v'] = away_adv_used

    cerebro = st.session_state.get('cerebro_ensemble')
    if cerebro is not None and st.session_state.get('tl_stats') is not None:
        try:
            tl = st.session_state['tl_stats']
            tv = st.session_state['tv_stats']
            tl['intPoints'] = 0
            tv['intPoints'] = 0

            features = calcular_caracteristicas_ensemble(
                tl=tl, tv=tv,
                prom_media_liga=st.session_state.get('prom_media_copa', 1.25),
                gf_rec_l=st.session_state.get('gf_rec_l'),
                gc_rec_l=st.session_state.get('gc_rec_l'),
                gf_rec_v=st.session_state.get('gf_rec_v'),
                gc_rec_v=st.session_state.get('gc_rec_v'),
                elo_l=st.session_state.get('elo_l', 1.0),
                elo_v=st.session_state.get('elo_v', 1.0),
                pit_l=st.session_state.get('pit_l', 0.5),
                pit_v=st.session_state.get('pit_v', 0.5),
                home_adv=home_adv_used,
                away_adv=away_adv_used,
                cuota_local=o1,
                cuota_empate=ox,
                cuota_visita=o2,
                gd_h_3=st.session_state.get('gd_h_3', 0),
                gd_a_3=st.session_state.get('gd_a_3', 0),
                gd_h_5=st.session_state.get('gd_h_5', 0),
                gd_a_5=st.session_state.get('gd_a_5', 0),
                gd_h_10=st.session_state.get('gd_h_10', 0),
                gd_a_10=st.session_state.get('gd_a_10', 0),
                avg_gf_h_3=st.session_state.get('avg_gf_h_3', 0.0),
                avg_gc_h_3=st.session_state.get('avg_gc_h_3', 0.0),
                avg_gf_a_3=st.session_state.get('avg_gf_a_3', 0.0),
                avg_gc_a_3=st.session_state.get('avg_gc_a_3', 0.0),
                btts_h_5=st.session_state.get('btts_h_5', 0.0),
                btts_a_5=st.session_state.get('btts_a_5', 0.0),
                win_streak_h=st.session_state.get('win_streak_h', 0),
                loss_streak_h=st.session_state.get('loss_streak_h', 0),
                win_streak_a=st.session_state.get('win_streak_a', 0),
                loss_streak_a=st.session_state.get('loss_streak_a', 0),
                racha_esp_local=st.session_state.get('racha_esp_local', 0.0),
                racha_esp_visita=st.session_state.get('racha_esp_visita', 0.0),
                volatilidad_local=st.session_state.get('volatilidad_local', 0.0),
                volatilidad_visita=st.session_state.get('volatilidad_visita', 0.0),
                momentum_local=st.session_state.get('momentum_local', 0.0),
                momentum_visita=st.session_state.get('momentum_visita', 0.0)
            )
            X_pred = pd.DataFrame([features])
            expected_cols = cerebro['feature_names']
            for col in expected_cols:
                if col not in X_pred.columns:
                    X_pred[col] = 0
            X_pred = X_pred[expected_cols]
            probas = cerebro['modelo_clasificacion'].predict_proba(X_pred)[0]
            prob_local, prob_empate, prob_visita = probas[0]*100, probas[1]*100, probas[2]*100

            draw_rate = st.session_state.get('draw_rate_auto', 0.25)
            motor = MotorMatematico(p_copa, draw_rate, liga_id=None, round_name=ronda)
            xg_l_adj = lgf
            xg_v_adj = vgf

            rest_local = st.session_state.get('rest_days_local', 7)
            rest_visitor = st.session_state.get('rest_days_visitor', 7)
            rest_factor = min(1.2, max(0.8, rest_local / max(1, rest_visitor)))
            xg_l_adj *= rest_factor
            xg_v_adj /= rest_factor

            if adjustments:
                xg_l_adj *= adjustments['xg_factor']
                xg_v_adj *= adjustments['xg_factor']

            if adjustments:
                prob_empate += adjustments['draw_boost'] * 100
                total_prob = prob_local + prob_empate + prob_visita
                prob_local = prob_local / total_prob * 100
                prob_empate = prob_empate / total_prob * 100
                prob_visita = prob_visita / total_prob * 100

            res = motor.procesar(xg_l_adj, xg_v_adj, cuotas=(o1, ox, o2), round_name=ronda)
            res['1X2'] = (prob_local, prob_empate, prob_visita)
            prob_reales = motor.desvig_odds((o1, ox, o2))
            res['EV'] = [((p/100)*c)-1 for p,c in zip([prob_local, prob_empate, prob_visita], (o1, ox, o2))]
            res['KELLY'] = [motor.calcular_kelly(p, c, pr) for p,c,pr in zip([prob_local, prob_empate, prob_visita], (o1, ox, o2), prob_reales)]
            res['DC'] = ((prob_local+prob_empate), (prob_visita+prob_empate), (prob_local+prob_visita))
            st.success(f"✅ Usando cerebro {st.session_state.get('cerebro_path', 'desconocido')}" + (" + Contexto Copa" if adjustments else ""))
        except Exception as e:
            st.warning(f"⚠️ Error usando cerebro: {e}. Usando motor matemático.")
            cerebro = None

    if res is None:
        draw_rate = st.session_state.get('draw_rate_auto', 0.25)
        motor = MotorMatematico(p_copa, draw_rate, liga_id=None, round_name=ronda)
        if modo_neutral:
            f_adv_l, f_adv_v = 1.0, 1.0
        else:
            f_adv_l = home_adv_used
            f_adv_v = away_adv_used
        xg_l_final = (lgf/p_copa)*(vgc/p_copa)*p_copa * f_adv_l
        xg_v_final = (vgf/p_copa)*(lgc/p_copa)*p_copa * f_adv_v

        rest_local = st.session_state.get('rest_days_local', 7)
        rest_visitor = st.session_state.get('rest_days_visitor', 7)
        rest_factor = min(1.2, max(0.8, rest_local / max(1, rest_visitor)))
        xg_l_final *= rest_factor
        xg_v_final /= rest_factor

        if adjustments:
            xg_l_final *= adjustments['xg_factor']
            xg_v_final *= adjustments['xg_factor']
        xg_l_final = max(0.3, min(4.5, xg_l_final))
        xg_v_final = max(0.3, min(4.5, xg_v_final))
        res = motor.procesar(xg_l_final, xg_v_final, cuotas=(o1, ox, o2), round_name=ronda)
        if adjustments:
            prob_local, prob_empate, prob_visita = res['1X2']
            prob_empate += adjustments['draw_boost'] * 100
            total_prob = prob_local + prob_empate + prob_visita
            if total_prob > 0:
                prob_local = prob_local / total_prob * 100
                prob_empate = prob_empate / total_prob * 100
                prob_visita = prob_visita / total_prob * 100
            res['1X2'] = (prob_local, prob_empate, prob_visita)
            prob_reales = motor.desvig_odds((o1, ox, o2))
            res['EV'] = [((p/100)*c)-1 for p,c in zip([prob_local, prob_empate, prob_visita], (o1, ox, o2))]
            res['KELLY'] = [motor.calcular_kelly(p, c, pr) for p,c,pr in zip([prob_local, prob_empate, prob_visita], (o1, ox, o2), prob_reales)]
            res['DC'] = ((prob_local+prob_empate), (prob_visita+prob_empate), (prob_local+prob_visita))
        st.info("ℹ️ Usando motor matemático" + (" + Contexto Copa" if adjustments else ""))

    if res is None:
        st.error("❌ No se pudo generar la predicción. Revisa los datos de entrada.")
        st.stop()

    # --- PANEL DE SUGERENCIAS DESTACADAS ---
    sugerencias = generar_sugerencias(res, (o1, ox, o2), (nl, nv))
    mostrar_panel_sugerencias(sugerencias)

    # Mostrar análisis de contexto si existe
    if adjustments:
        with st.expander("🏆 Análisis de Contexto de Copa", expanded=True):
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("🎯 Importancia", f"{adjustments['importance']*100:.0f}%")
                st.progress(adjustments['importance'], text=f"Nivel: {adjustments['level']}")
            with col2:
                st.markdown("**Factores detectados:**")
                for factor, value in adjustments['factors'].items():
                    if value > 0:
                        st.markdown(f"- {factor}: {value*100:.0f}%")
            st.markdown(f"**Ajustes:** xG × {adjustments['xg_factor']:.2f} | Empate +{adjustments['draw_boost']*100:.0f}%")

    # Pestañas de resultados
    t1, t2, t3, t4, t5 = st.tabs(["🥅 Goles", "📊 1X2", "🛡️ Doble O", "🎲 Simulador", "🧩 Matriz"])

    with t1:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        col_g1, col_g2 = st.columns(2)
        for i, line in enumerate([0.5, 1.5, 2.5, 3.5, 4.5, 5.5]):
            p_o, p_u = res['GOLES'][line]
            with (col_g1 if i < 3 else col_g2):
                render_dual_bar(f"Línea Total {line}", p_o, p_u)
        st.divider()
        render_dual_bar("Ambos Anotan (BTTS)", res['BTTS'][0], res['BTTS'][1], c_over="var(--primary)", c_under="var(--text-sub)")
        st.markdown('</div>', unsafe_allow_html=True)

    with t2:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        o_c1, o_c2, o_c3 = st.columns(3)
        with o_c1: render_outcome_card(f"Local", res['1X2'][0], ev=res['EV'][0], kelly=res['KELLY'][0], color="var(--primary)")
        with o_c2: render_outcome_card("Empate", res['1X2'][1], ev=res['EV'][1], kelly=res['KELLY'][1], color="var(--text-sub)")
        with o_c3: render_outcome_card(f"Visita", res['1X2'][2], ev=res['EV'][2], kelly=res['KELLY'][2], color="var(--emerald)")
        st.markdown('</div>', unsafe_allow_html=True)

    with t3:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        dc1, dc2, dc3 = st.columns(3)
        with dc1: render_outcome_card("Local o Empate (1X)", res['DC'][0], color="var(--primary)")
        with dc2: render_outcome_card("Visita o Empate (X2)", res['DC'][1], color="var(--emerald)")
        with dc3: render_outcome_card("Cualquiera (12)", res['DC'][2], color="#fff")
        st.markdown('</div>', unsafe_allow_html=True)

    with t4:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Simulación Estocástica (100k)</div>', unsafe_allow_html=True)
        mc = res['MONTECARLO']
        raw_h = mc['RAW_H']
        raw_v = mc['RAW_V']
        tot_sim = mc['RAW_TOTALS']
        margin = raw_h - raw_v

        ah_l = {
            "-0.5": (margin >= 1).mean() * 100,
            "-1.5": (margin >= 2).mean() * 100,
            "-2.5": (margin >= 3).mean() * 100,
            "-3.5": (margin >= 4).mean() * 100,
            "+0.5": (margin >= 0).mean() * 100,
            "+1.5": (margin >= -1).mean() * 100,
            "+2.5": (margin >= -2).mean() * 100,
            "+3.5": (margin >= -3).mean() * 100,
        }
        ah_v = {
            "-0.5": (margin <= -1).mean() * 100,
            "-1.5": (margin <= -2).mean() * 100,
            "-2.5": (margin <= -3).mean() * 100,
            "-3.5": (margin <= -4).mean() * 100,
            "+0.5": (margin <= 0).mean() * 100,
            "+1.5": (margin <= 1).mean() * 100,
            "+2.5": (margin <= 2).mean() * 100,
            "+3.5": (margin <= 3).mean() * 100,
        }
        o_15 = (tot_sim >= 2).mean() * 100
        u_15 = 100 - o_15
        o_35 = (tot_sim >= 4).mean() * 100
        u_35 = 100 - o_35
        win_nil_l = ((raw_h > raw_v) & (raw_v == 0)).mean() * 100
        win_nil_v = ((raw_v > raw_h) & (raw_h == 0)).mean() * 100
        btts_o25 = ((raw_h > 0) & (raw_v > 0) & (tot_sim >= 3)).mean() * 100
        g_2_to_3 = ((tot_sim >= 2) & (tot_sim <= 3)).mean() * 100

        def make_card(label, val, color):
            return f'<div class="mc-mini-card"><div class="mc-label">{label}</div><div class="mc-val" style="color:{color};">{val:.1f}%</div></div>'

        st.markdown('<div style="font-size:0.75rem; color:var(--text-sub); font-weight:700; margin-bottom:8px;">HÁNDICAP ASIÁTICO LOCAL</div>', unsafe_allow_html=True)
        cards_ah_l = [make_card(k, v, "var(--primary)") for k, v in ah_l.items()]
        st.markdown(f'<div class="mc-grid">{"".join(cards_ah_l)}</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.75rem; color:var(--text-sub); font-weight:700; margin-bottom:8px;">HÁNDICAP ASIÁTICO VISITANTE</div>', unsafe_allow_html=True)
        cards_ah_v = [make_card(k, v, "var(--emerald)") for k, v in ah_v.items()]
        st.markdown(f'<div class="mc-grid">{"".join(cards_ah_v)}</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.75rem; color:var(--text-sub); font-weight:700; margin-bottom:8px;">TOTALES & PROPS ESPECIALES</div>', unsafe_allow_html=True)
        cards_mix = [
            make_card("Over 1.5", o_15, "var(--emerald)"),
            make_card("Under 1.5", u_15, "var(--ruby)"),
            make_card("Over 3.5", o_35, "var(--emerald)"),
            make_card("Under 3.5", u_35, "var(--ruby)"),
            make_card("Ambos + O2.5", btts_o25, "var(--primary)"),
            make_card("Goles 2 a 3", g_2_to_3, "#fff"),
            make_card("Gana L a 0", win_nil_l, "var(--primary)"),
            make_card("Gana V a 0", win_nil_v, "var(--emerald)")
        ]
        st.markdown(f'<div class="mc-grid">{"".join(cards_mix)}</div>', unsafe_allow_html=True)

        st.markdown("<div style='margin-top:20px; border-top:1px solid var(--border-color); padding-top:15px;'></div>", unsafe_allow_html=True)
        g1, g2 = st.columns(2)
        with g1:
            win_2plus_l = (margin >= 2).mean() * 100
            win_1_l = (margin == 1).mean() * 100
            draw = (margin == 0).mean() * 100
            win_1_v = (margin == -1).mean() * 100
            win_2plus_v = (margin <= -2).mean() * 100
            labels_margin = ["L por 2+", "L por 1", "Empate", "V por 1", "V por 2+"]
            vals_margin = [win_2plus_l, win_1_l, draw, win_1_v, win_2plus_v]
            colors_m = ['#3B82F6', '#60A5FA', '#94A3B8', '#34D399', '#10B981']
            fig_pie = go.Figure(data=[go.Pie(labels=labels_margin, values=vals_margin, hole=.4, marker_colors=colors_m, textinfo='label+percent')])
            fig_pie.update_layout(title="Margen de Victoria Exacto", title_font_size=13, title_x=0.5, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#94A3B8'), margin=dict(l=0, r=0, t=30, b=0), height=220, showlegend=False)
            st.plotly_chart(fig_pie, use_container_width=True)
        with g2:
            totals_count = pd.Series(tot_sim).value_counts(normalize=True).sort_index() * 100
            totals_count = totals_count[totals_count.index <= 6]
            colors_t = ['#EF4444' if x < 2.5 else '#10B981' for x in totals_count.index]
            fig_ou = go.Figure()
            fig_ou.add_trace(go.Bar(x=totals_count.index, y=totals_count.values, marker_color=colors_t, text=[f"{v:.1f}%" for v in totals_count.values], textposition='outside', textfont=dict(color='#fff', size=11)))
            fig_ou.update_layout(title="Riesgo O/U 2.5 (Verde=Over)", title_font_size=13, title_x=0.5, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#94A3B8'), xaxis=dict(showgrid=False, tickmode='linear'), yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', visible=False, range=[0, totals_count.max() * 1.3]), margin=dict(l=0, r=0, t=30, b=0), height=220)
            st.plotly_chart(fig_ou, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with t5:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        fig_mat = px.imshow(res['MATRIZ'], text_auto=".1f", color_continuous_scale='Blues', labels=dict(x="Goles Visitante", y="Goles Local"))
        fig_mat.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#fff")
        st.plotly_chart(fig_mat, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
