# app_cup.py — Calculadora de predicción de partidos con xG manuales
import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import logging

from math_engine import MotorMatematico
from visual_components import render_dual_bar, render_outcome_card, apply_custom_css

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generar_sugerencias(res, cuotas, nombres_equipos):
    sugerencias = []
    prob_local, prob_empate, prob_visita = res['1X2']
    ev_local, ev_empate, ev_visita = res['EV']
    kelly_local, kelly_empate, kelly_visita = res['KELLY']
    o1, ox, o2 = cuotas

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

    html_parts = ['<div class="suggestions-panel"><div class="suggestions-title">🎯 Sugerencias Destacadas <span class="suggestion-badge">ALTA CONFIANZA</span></div><div class="suggestion-grid">']

    for sug in sugerencias:
        mercado = sug['mercado']
        seleccion = sug['seleccion']
        razon = sug['razon']
        prob = f"{sug['prob']:.1f}"
        ev_str = f" · EV {sug['ev']:+.2f}" if sug.get('ev') is not None else ''
        kelly_str = f" · K {sug['kelly']:.1f}%" if sug.get('kelly') is not None and sug['kelly'] > 0 else ''
        cuota_str = f" · Cuota {sug['cuota']:.2f}" if sug.get('cuota') is not None else ''
        metricas = f"{ev_str}{kelly_str}{cuota_str}"

        card = (
            f'<div class="suggestion-card">'
            f'<div class="suggestion-mercado">{mercado}</div>'
            f'<div class="suggestion-seleccion">{seleccion}</div>'
            f'<div style="display: flex; align-items: baseline; gap: 8px;">'
            f'<span class="suggestion-prob">{prob}%</span>'
            f'<span class="suggestion-ev">{metricas}</span>'
            f'</div>'
            f'<div class="suggestion-razon">{razon}</div>'
            f'</div>'
        )
        html_parts.append(card)

    html_parts.append('</div></div>')
    full_html = ''.join(html_parts)
    st.markdown(full_html, unsafe_allow_html=True)

# ============================================================
# INTERFAZ DE USUARIO
# ============================================================
apply_custom_css()
st.set_page_config(page_title="DeepXG Cup Predictor", layout="wide")

st.markdown("<h1 class='app-title'>DeepXG <span>Cup Predictor</span></h1>", unsafe_allow_html=True)
st.markdown("<p class='app-subtitle'>PREDICCIÓN BASADA EN xG MANUALES</p>", unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">⚙️ Parámetros del Partido</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        nl = st.text_input("🏠 Nombre Local", "Local")
        lgf = st.number_input("xG Local", 0.0, 5.0, 1.2, step=0.05)
    with col2:
        nv = st.text_input("✈️ Nombre Visitante", "Visitante")
        vgf = st.number_input("xG Visitante", 0.0, 5.0, 1.0, step=0.05)

    p_copa = st.number_input("Promedio de goles de la liga/copa", 0.5, 5.0, 2.5, step=0.01)

    st.markdown("#### 📊 Cuotas de Mercado (1X2)")
    odd_cols = st.columns(3)
    with odd_cols[0]:
        o1 = st.number_input("Local (1)", 1.0, 50.0, 2.0, step=0.1)
    with odd_cols[1]:
        ox = st.number_input("Empate (X)", 1.0, 50.0, 3.4, step=0.1)
    with odd_cols[2]:
        o2 = st.number_input("Visita (2)", 1.0, 50.0, 3.2, step=0.1)

    analizar_btn = st.button("🚀 PROCESAR PREDICCIÓN", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

if analizar_btn:
    # Instanciar motor con el promedio de goles manual
    motor = MotorMatematico(p_copa, draw_rate_real=0.25)
    # Procesar directamente con los xG manuales y las cuotas
    res = motor.procesar(lgf, vgf, cuotas=(o1, ox, o2))

    # Sugerencias
    sugerencias = generar_sugerencias(res, (o1, ox, o2), (nl, nv))
    mostrar_panel_sugerencias(sugerencias)

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