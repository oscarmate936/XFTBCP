import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from fuzzywuzzy import process
from collections import Counter
import joblib
import os
from datetime import datetime
import pytz
import logging

from math_engine import MotorMatematico
from api_utils import call_api, get_cup_matches, get_team_cup_stats
from data_processing import calc_fuerza_pitagorica, calc_elo, calcular_stats_avanzadas_copa, resolver_colley_simple
from visual_components import render_dual_bar, render_outcome_card, apply_custom_css
from cup_context_analyzer import CupContextAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# INICIALIZACIÓN DEL ESTADO DE SESIÓN
# ============================================================
if 'p_copa_auto' not in st.session_state:
    st.session_state['p_copa_auto'] = 2.5
if 'nl_auto' not in st.session_state:
    st.session_state['nl_auto'] = "Equipo Local"
if 'nv_auto' not in st.session_state:
    st.session_state['nv_auto'] = "Equipo Visitante"
if 'h_adv_l' not in st.session_state:
    st.session_state['h_adv_l'] = 1.10
if 'v_adv_v' not in st.session_state:
    st.session_state['v_adv_v'] = 0.90
if 'cup_matches_cached' not in st.session_state:
    st.session_state['cup_matches_cached'] = []
if 'draw_rate_auto' not in st.session_state:
    st.session_state['draw_rate_auto'] = 0.25
if 'tl_stats' not in st.session_state:
    st.session_state['tl_stats'] = None
if 'tv_stats' not in st.session_state:
    st.session_state['tv_stats'] = None
if 'lgf_auto' not in st.session_state:
    st.session_state['lgf_auto'] = 1.2
if 'lgc_auto' not in st.session_state:
    st.session_state['lgc_auto'] = 1.0
if 'vgf_auto' not in st.session_state:
    st.session_state['vgf_auto'] = 1.1
if 'vgc_auto' not in st.session_state:
    st.session_state['vgc_auto'] = 1.3
if 'prom_media_copa' not in st.session_state:
    st.session_state['prom_media_copa'] = 1.25

# Variables para features del cerebro
for key in ['gd_h_3', 'gd_a_3', 'gd_h_5', 'gd_a_5', 'gd_h_10', 'gd_a_10',
            'avg_gf_h_3', 'avg_gc_h_3', 'avg_gf_a_3', 'avg_gc_a_3',
            'btts_h_5', 'btts_a_5', 'win_streak_h', 'loss_streak_h',
            'win_streak_a', 'loss_streak_a', 'racha_esp_local', 'racha_esp_visita',
            'volatilidad_local', 'volatilidad_visita', 'momentum_local', 'momentum_visita',
            'elo_l', 'elo_v', 'pit_l', 'pit_v']:
    if key not in st.session_state:
        st.session_state[key] = 0 if 'streak' in key or 'gd_' in key else 0.0

# ============================================================
# CARGA DEL CEREBRO
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
        if 'cup' in cerebro_path:
            st.success("🧠 Usando cerebro especializado en COPAS")
        else:
            st.info("🧠 Usando cerebro general (ligas) - puede ser menos preciso en copas")
    except Exception as e:
        st.session_state['cerebro_ensemble'] = None
        st.warning(f"Error cargando cerebro ({cerebro_path}): {e}")
else:
    st.session_state['cerebro_ensemble'] = None
    st.info("ℹ️ No se encontró archivo de cerebro. Se usará el motor matemático.")

apply_custom_css()
st.set_page_config(page_title="DeepXG Cup Predictor", layout="wide", initial_sidebar_state="expanded")

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================
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
    meses = ['enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio', 'julio', 'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre']
    return f"{dias[dt.weekday()]} {dt.day} de {meses[dt.month-1]}"

def formatear_hora_local(dt):
    return dt.strftime("%H:%M") if dt else "??:??"

def obtener_estadisticas_avanzadas_copa(team_id, cup_matches, max_partidos=10, prom_media_liga=1.25):
    team_matches = []
    for ev in cup_matches:
        try:
            if str(ev.get('idHomeTeam')) == str(team_id) or str(ev.get('idAwayTeam')) == str(team_id):
                if ev.get('intHomeScore') is not None:
                    team_matches.append(ev)
        except:
            continue
    team_matches.sort(key=lambda x: x['dateEvent'])
    if not team_matches:
        return (0, 0, 0, 0.0, 0.0, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    gf_list = []
    gc_list = []
    pts_list = []
    fechas = []
    
    for ev in team_matches[:max_partidos]:
        try:
            fecha = datetime.strptime(ev['dateEvent'], '%Y-%m-%d')
            s_h = int(ev['intHomeScore'])
            s_a = int(ev['intAwayScore'])
            if str(ev.get('idHomeTeam')) == str(team_id):
                gf_list.append(s_h)
                gc_list.append(s_a)
                pts_list.append(3 if s_h > s_a else (1 if s_h == s_a else 0))
            else:
                gf_list.append(s_a)
                gc_list.append(s_h)
                pts_list.append(3 if s_a > s_h else (1 if s_a == s_h else 0))
            fechas.append(fecha)
        except:
            continue
    
    if not gf_list:
        return (0, 0, 0, 0.0, 0.0, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
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
    racha = sum(p * w for p, w in zip(pts_list[:5], pesos[:len(pts_list[:5])])) if n >= 5 else np.mean(pts_list) if pts_list else 1.5
    
    xg, vol, mom = calcular_stats_avanzadas_copa(gf_list, fechas, prom_media_liga)
    def_xg, _, _ = calcular_stats_avanzadas_copa(gc_list, fechas, prom_media_liga)
    
    return (gd_3, gd_5, gd_10, avg_gf_3, avg_gc_3, btts_ratio,
            win_streak, loss_streak, racha, vol, mom, xg, def_xg)

def calcular_caracteristicas_cerebro_cup(team_id_local, team_id_visitante, cup_matches, prom_media_copa):
    (gd_h_3, gd_h_5, gd_h_10, avg_gf_h_3, avg_gc_h_3, btts_h_5,
     win_streak_h, loss_streak_h, racha_h, vol_h, mom_h, xg_l, def_l) = obtener_estadisticas_avanzadas_copa(
         team_id_local, cup_matches, prom_media_liga=prom_media_copa)
    (gd_a_3, gd_a_5, gd_a_10, avg_gf_a_3, avg_gc_a_3, btts_a_5,
     win_streak_a, loss_streak_a, racha_a, vol_a, mom_a, xg_v, def_v) = obtener_estadisticas_avanzadas_copa(
         team_id_visitante, cup_matches, prom_media_liga=prom_media_copa)
    
    all_teams = set()
    for ev in cup_matches:
        all_teams.add(ev['strHomeTeam'])
        all_teams.add(ev['strAwayTeam'])
    team_to_idx = {team: i for i, team in enumerate(all_teams)}
    partidos_colley = []
    for ev in cup_matches:
        try:
            h = ev['strHomeTeam']
            v = ev['strAwayTeam']
            gh = int(ev['intHomeScore'])
            ga = int(ev['intAwayScore'])
            resultado = 0 if gh > ga else (1 if gh == ga else 2)
            partidos_colley.append((team_to_idx[h], team_to_idx[v], resultado))
        except:
            continue
    colley_ratings = resolver_colley_simple(partidos_colley, team_to_idx)
    rating_local = colley_ratings.get(st.session_state['nl_auto'], 0.5)
    rating_visit = colley_ratings.get(st.session_state['nv_auto'], 0.5)
    
    ventaja_local = xg_l * (rating_local + 0.5)
    ventaja_visita = xg_v * (rating_visit + 0.5)
    
    pj_l, gf_l, gc_l = get_team_cup_stats(team_id_local, cup_matches)
    pj_v, gf_v, gc_v = get_team_cup_stats(team_id_visitante, cup_matches)
    max_pts = max(pj_l*3, pj_v*3)
    elo_l = calc_elo(gf_l - gc_l, max_pts, 0) if pj_l > 0 else 1.0
    elo_v = calc_elo(gf_v - gc_v, max_pts, 0) if pj_v > 0 else 1.0
    elo_diff = elo_l - elo_v
    
    pit_l = calc_fuerza_pitagorica(gf_l, gc_l) if pj_l > 0 else 0.5
    pit_v = calc_fuerza_pitagorica(gf_v, gc_v) if pj_v > 0 else 0.5
    
    features = {
        'ventaja_local': ventaja_local,
        'ventaja_visita': ventaja_visita,
        'defensa_local': def_l,
        'defensa_visita': def_v,
        'elo_diff': elo_diff,
        'racha_esp_local': racha_h,
        'racha_esp_visita': racha_a,
        'volatilidad_local': vol_h,
        'volatilidad_visita': vol_a,
        'momentum_local': mom_h,
        'momentum_visita': mom_a,
        'gd_h_3': gd_h_3, 'gd_a_3': gd_a_3,
        'gd_h_5': gd_h_5, 'gd_a_5': gd_a_5,
        'gd_h_10': gd_h_10, 'gd_a_10': gd_a_10,
        'avg_gf_h_3': avg_gf_h_3, 'avg_gc_h_3': avg_gc_h_3,
        'avg_gf_a_3': avg_gf_a_3, 'avg_gc_a_3': avg_gc_a_3,
        'btts_h_5': btts_h_5, 'btts_a_5': btts_a_5,
        'win_streak_h': win_streak_h, 'loss_streak_h': loss_streak_h,
        'win_streak_a': win_streak_a, 'loss_streak_a': loss_streak_a,
    }
    return features, (pit_l, pit_v)

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
        "🌍 Mundial de Clubes": {"id": 4488, "season_default": "2025"},
    }

    copa_sel = st.selectbox("Elegir Copa", list(copas.keys()))
    cup_info = copas[copa_sel]
    cup_id = cup_info["id"]
    season = st.text_input("Temporada (ej. 2025-2026 o 2025)", value=cup_info["season_default"])

    if st.button("🔄 Sincronizar Datos", use_container_width=True):
        with st.spinner(f"Obteniendo partidos de {copa_sel} temporada {season}..."):
            matches = get_cup_matches(cup_id, season)
        if matches:
            # Filtrar partidos con marcador válido para cálculos
            valid_matches = []
            total_goles = 0
            total_partidos = 0
            for m in matches:
                try:
                    if m.get('intHomeScore') is not None and m.get('intAwayScore') is not None:
                        gh = int(m['intHomeScore'])
                        ga = int(m['intAwayScore'])
                        total_goles += gh + ga
                        total_partidos += 1
                        valid_matches.append(m)
                except:
                    continue
            if total_partidos == 0:
                st.error("No se encontraron partidos con resultados válidos.")
            else:
                st.session_state['cup_matches_cached'] = valid_matches
                prom_goles = total_goles / total_partidos
                st.session_state['p_copa_auto'] = round(prom_goles, 2)
                # Calcular draw_rate real de la copa
                empates = sum(1 for m in valid_matches if int(m['intHomeScore']) == int(m['intAwayScore']))
                st.session_state['draw_rate_auto'] = empates / total_partidos
                st.success(f"✅ {len(valid_matches)} partidos sincronizados. Promedio goles: {prom_goles:.2f} | Empates: {st.session_state['draw_rate_auto']:.2%}")
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
                    st.session_state['prom_media_copa'] = prom_media
                    
                    max_pts = max(pj_l*3, pj_v*3)
                    elo_l = calc_elo(gf_l - gc_l, max_pts, 0)
                    elo_v = calc_elo(gf_v - gc_v, max_pts, 0)
                    pit_l = calc_fuerza_pitagorica(gf_l, gc_l)
                    pit_v = calc_fuerza_pitagorica(gf_v, gc_v)
                    home_adv = 1.1
                    away_adv = 0.9
                    xg_l_base = (gf_l / max(pj_l,1)) * elo_l * pit_l
                    xg_v_base = (gf_v / max(pj_v,1)) * elo_v * pit_v
                    st.session_state['lgf_auto'] = xg_l_base * home_adv
                    st.session_state['vgf_auto'] = xg_v_base * away_adv
                    st.session_state['lgc_auto'] = (gc_l / max(pj_l,1)) / max(elo_v, 0.1)
                    st.session_state['vgc_auto'] = (gc_v / max(pj_v,1)) / max(elo_l, 0.1)
                    
                    features, (pit_l_val, pit_v_val) = calcular_caracteristicas_cerebro_cup(
                        m['idHomeTeam'], m['idAwayTeam'], st.session_state['cup_matches_cached'], prom_media)
                    for k, v in features.items():
                        if k in st.session_state:
                            st.session_state[k] = v
                    st.session_state['elo_l'] = elo_l
                    st.session_state['elo_v'] = elo_v
                    st.session_state['pit_l'] = pit_l_val
                    st.session_state['pit_v'] = pit_v_val
                    st.rerun()
            else:
                st.info("No hay partidos programados en los próximos 2 días para esta competición.")
                with st.expander("Ver todos los partidos sincronizados"):
                    try:
                        all_sorted = sorted(all_matches, key=lambda x: x.get('dateEvent', ''))
                    except:
                        all_sorted = all_matches
                    for ev in all_sorted[:20]:
                        dt_local = convertir_hora_elsalvador(ev.get('dateEvent', ''), ev.get('strTime', ''))
                        if dt_local:
                            fecha_str = formatear_dia_local(dt_local)
                            hora_str = formatear_hora_local(dt_local)
                            st.write(f"{fecha_str} {hora_str} - {ev['strHomeTeam']} vs {ev['strAwayTeam']}")
                        else:
                            st.write(f"{ev['dateEvent']} - {ev['strHomeTeam']} vs {ev['strAwayTeam']}")
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
            col_ida1, col_ida2 = st.columns(2)
            with col_ida1:
                goles_ida_local = st.number_input("Goles Local (ida)", 0, 10, 0)
            with col_ida2:
                goles_ida_visitante = st.number_input("Goles Visitante (ida)", 0, 10, 0)
            first_leg_result = (goles_ida_local, goles_ida_visitante)
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

    cerebro = st.session_state.get('cerebro_ensemble')
    if cerebro is not None and st.session_state.get('cup_matches_cached') and st.session_state.get('tl_stats') is not None:
        try:
            if 'ventaja_local' not in st.session_state:
                st.warning("No se han cargado estadísticas previas. Por favor, selecciona un partido de la lista y haz clic en 'Cargar Estadísticas'.")
                cerebro = None
            else:
                features = {
                    'ventaja_local': st.session_state['ventaja_local'],
                    'ventaja_visita': st.session_state['ventaja_visita'],
                    'defensa_local': st.session_state['defensa_local'],
                    'defensa_visita': st.session_state['defensa_visita'],
                    'elo_diff': st.session_state['elo_l'] - st.session_state['elo_v'],
                    'racha_esp_local': st.session_state['racha_esp_local'],
                    'racha_esp_visita': st.session_state['racha_esp_visita'],
                    'volatilidad_local': st.session_state['volatilidad_local'],
                    'volatilidad_visita': st.session_state['volatilidad_visita'],
                    'momentum_local': st.session_state['momentum_local'],
                    'momentum_visita': st.session_state['momentum_visita'],
                    'gd_h_3': st.session_state['gd_h_3'], 'gd_a_3': st.session_state['gd_a_3'],
                    'gd_h_5': st.session_state['gd_h_5'], 'gd_a_5': st.session_state['gd_a_5'],
                    'gd_h_10': st.session_state['gd_h_10'], 'gd_a_10': st.session_state['gd_a_10'],
                    'avg_gf_h_3': st.session_state['avg_gf_h_3'], 'avg_gc_h_3': st.session_state['avg_gc_h_3'],
                    'avg_gf_a_3': st.session_state['avg_gf_a_3'], 'avg_gc_a_3': st.session_state['avg_gc_a_3'],
                    'btts_h_5': st.session_state['btts_h_5'], 'btts_a_5': st.session_state['btts_a_5'],
                    'win_streak_h': st.session_state['win_streak_h'], 'loss_streak_h': st.session_state['loss_streak_h'],
                    'win_streak_a': st.session_state['win_streak_a'], 'loss_streak_a': st.session_state['loss_streak_a'],
                }
                inv_odds = [1 / max(c, 1.01) for c in [o1, ox, o2]]
                prob_impl = [p / sum(inv_odds) for p in inv_odds]
                features['prob_impl_local'] = prob_impl[0]
                features['prob_impl_empate'] = prob_impl[1]
                features['prob_impl_visita'] = prob_impl[2]
                
                X_pred = pd.DataFrame([features])
                expected_cols = cerebro['feature_names']
                for col in expected_cols:
                    if col not in X_pred.columns:
                        X_pred[col] = 0
                X_pred = X_pred[expected_cols]
                probas = cerebro['modelo_clasificacion'].predict_proba(X_pred)[0]
                prob_local, prob_empate, prob_visita = probas[0]*100, probas[1]*100, probas[2]*100
                
                if adjustments:
                    prob_empate += adjustments['draw_boost'] * 100
                    total_prob = prob_local + prob_empate + prob_visita
                    if total_prob > 0:
                        prob_local = prob_local / total_prob * 100
                        prob_empate = prob_empate / total_prob * 100
                        prob_visita = prob_visita / total_prob * 100
                
                draw_rate = st.session_state.get('draw_rate_auto', 0.25)
                motor = MotorMatematico(p_copa, draw_rate, liga_id=None)
                xg_l_adj = lgf
                xg_v_adj = vgf
                if adjustments:
                    xg_l_adj *= adjustments['xg_factor']
                    xg_v_adj *= adjustments['xg_factor']
                res = motor.procesar(xg_l_adj, xg_v_adj, cuotas=(o1, ox, o2))
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
        motor = MotorMatematico(p_copa, draw_rate, liga_id=None)
        if modo_neutral:
            f_adv_l, f_adv_v = 1.0, 1.0
        else:
            f_adv_l = st.session_state.get('h_adv_l', 1.1)
            f_adv_v = st.session_state.get('v_adv_v', 0.9)
        xg_l_final = lgf * f_adv_l
        xg_v_final = vgf * f_adv_v
        if adjustments:
            xg_l_final *= adjustments['xg_factor']
            xg_v_final *= adjustments['xg_factor']
        xg_l_final = max(0.3, min(4.5, xg_l_final))
        xg_v_final = max(0.3, min(4.5, xg_v_final))
        res = motor.procesar(xg_l_final, xg_v_final, cuotas=(o1, ox, o2))
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

    t1, t2, t3, t4, t5, t6 = st.tabs(["🥅 Goles", "📊 1X2", "🛡️ Doble O", "🎲 Simulador", "🧩 Matriz", "🕵️ Backtest"])

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

    with t6:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">🕵️ Backtest</div>', unsafe_allow_html=True)
        st.info("El backtest para copas requiere partidos históricos de la misma competición. Sincroniza los datos y luego implementa la lógica similar a la versión de liga.")
        st.markdown('</div>', unsafe_allow_html=True)