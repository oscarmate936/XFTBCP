# app.py
# DeepXG Cup Predictor - Versión con sugerencias automáticas y sin backtest
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
# FUNCIONES AUXILIARES PARA SUGERENCIAS INTELIGENTES
# ============================================================
def generar_sugerencias(res, cuotas, nombres_equipos):
    """
    Analiza los resultados del modelo y genera una lista de sugerencias
    priorizadas por probabilidad, valor esperado y Kelly.
    Retorna una lista de diccionarios con: 'mercado', 'seleccion', 'prob', 'cuota', 'ev', 'kelly', 'razon'.
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
    # Local -0.5
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
    # Visitante +0.5
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

    # Ordenar por probabilidad descendente y luego por EV
    sugerencias.sort(key=lambda x: (x.get('ev') or -999, x['prob']), reverse=True)
    return sugerencias[:5]  # Mostrar máximo 5 sugerencias

def mostrar_panel_sugerencias(sugerencias):
    """Renderiza un panel atractivo con las sugerencias destacadas."""
    if not sugerencias:
        return
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

    html = '<div class="suggestions-panel">'
    html += '<div class="suggestions-title">🎯 Sugerencias Destacadas <span class="suggestion-badge">ALTA CONFIANZA</span></div>'
    html += '<div class="suggestion-grid">'
    for sug in sugerencias:
        ev_str = f' · EV {sug["ev"]:+.2f}' if sug.get('ev') is not None else ''
        kelly_str = f' · K {sug["kelly"]:.1f}%' if sug.get('kelly') is not None and sug["kelly"] > 0 else ''
        cuota_str = f' · Cuota {sug["cuota"]:.2f}' if sug.get('cuota') is not None else ''
        html += f'''
        <div class="suggestion-card">
            <div class="suggestion-mercado">{sug['mercado']}</div>
            <div class="suggestion-seleccion">{sug['seleccion']}</div>
            <div style="display: flex; align-items: baseline; gap: 8px;">
                <span class="suggestion-prob">{sug['prob']:.1f}%</span>
                <span class="suggestion-ev">{ev_str}{kelly_str}{cuota_str}</span>
            </div>
            <div class="suggestion-razon">{sug['razon']}</div>
        </div>
        '''
    html += '</div></div>'
    st.markdown(html, unsafe_allow_html=True)

# ============================================================
# (Resto de funciones: get_auto_home_away_adv, convertir_hora, etc.)
# ============================================================
# ... (copiar aquí el resto del código exactamente igual que en la versión anterior,
#      desde get_auto_home_away_adv hasta la parte de UI principal, 
#      pero reemplazando la sección de tabs para que solo tenga 5 pestañas)

# NOTA: Para mantener esta respuesta concisa, a continuación solo incluyo las partes
# modificadas del código final. Se asume que el resto de funciones (get_auto_home_away_adv,
# convertir_hora_elsalvador, calcular_caracteristicas_ensemble, obtener_estadisticas_avanzadas,
# find_first_leg_match, inicialización de session state, carga de cerebro, sidebar, etc.)
# permanecen idénticas a la versión completa proporcionada anteriormente.

# La única diferencia relevante está al final, en la parte de análisis y visualización.

# ============================================================
# (Código previo sin cambios hasta el bloque "if analizar_btn:")
# ============================================================

if analizar_btn:
    res = None
    adjustments = None

    # ... (análisis de contexto, ventaja local, uso de cerebro, motor matemático, etc.
    #      exactamente igual que en la versión anterior)

    # Después de obtener 'res' y antes de mostrar los tabs:

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

    # Pestañas (ahora solo 5, sin Backtest)
    t1, t2, t3, t4, t5 = st.tabs(["🥅 Goles", "📊 1X2", "🛡️ Doble O", "🎲 Simulador", "🧩 Matriz"])

    with t1:
        # ... contenido igual que antes ...
        pass
    with t2:
        # ... contenido igual que antes ...
        pass
    with t3:
        # ... contenido igual que antes ...
        pass
    with t4:
        # ... contenido igual que antes ...
        pass
    with t5:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        fig_mat = px.imshow(res['MATRIZ'], text_auto=".1f", color_continuous_scale='Blues',
                            labels=dict(x="Goles Visitante", y="Goles Local"))
        fig_mat.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#fff")
        st.plotly_chart(fig_mat, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
