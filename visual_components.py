# visual_components.py
import streamlit as st

def render_dual_bar(label, p_over, p_under, c_over="var(--primary)", c_under="var(--ruby)"):
    st.markdown(
        f'''
        <div class="market-bar-container">
            <div class="market-bar-header">
                <span class="market-bar-label">{label}</span>
                <div class="market-bar-percentages">
                    <span style="color:{c_over};">{p_over:.1f}%</span>
                    <span style="color:var(--text-sub);">|</span>
                    <span style="color:{c_under};">{p_under:.1f}%</span>
                </div>
            </div>
            <div class="market-bar-track">
                <div class="market-bar-fill over" style="width:{p_over}%; background:{c_over};"></div>
                <div class="market-bar-fill under" style="width:{p_under}%; background:{c_under}; opacity:0.5;"></div>
            </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

def render_outcome_card(label, prob, ev=None, kelly=0, color="var(--emerald)"):
    ev_html = f'<div class="outcome-ev" style="color:{color};">EV {ev:+.2f}</div>' if ev is not None else ""
    k_html = f'<div class="outcome-kelly">STAKE {kelly:.1f}%</div>' if kelly > 0 else ""
    html = '<div class="outcome-card" style="border-bottom: 3px solid ' + color + ';">'
    html += '<div class="outcome-label">' + label + '</div>'
    html += '<div class="outcome-prob">' + f"{prob:.1f}%" + '</div>'
    html += '<div class="outcome-metrics">' + ev_html + k_html + '</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

def apply_custom_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&display=swap');
        
        * { box-sizing: border-box; }
        
        html, body, .stApp {
            font-family: 'Inter', sans-serif;
            background: #0B1120;
            color: #F1F5F9;
        }
        
        :root {
            --bg-primary: #0B1120;
            --bg-card: #1E293B;
            --bg-card-hover: #2D3A4E;
            --primary: #3B82F6;
            --primary-dark: #2563EB;
            --emerald: #10B981;
            --ruby: #EF4444;
            --amber: #F59E0B;
            --text-main: #F8FAFC;
            --text-sub: #94A3B8;
            --border-light: rgba(255,255,255,0.08);
            --shadow-sm: 0 4px 6px rgba(0,0,0,0.3);
            --shadow-md: 0 8px 12px rgba(0,0,0,0.4);
            --radius-md: 16px;
            --radius-lg: 24px;
        }
        
        .stApp { background: var(--bg-primary); padding: 0.5rem 0.75rem 1.5rem 0.75rem; }
        header[data-testid="stHeader"] { background: transparent !important; backdrop-filter: blur(8px); border-bottom: 1px solid var(--border-light); }
        section[data-testid="stSidebar"] { background: #0F172A; border-right: 1px solid var(--border-light); padding-top: 1rem; }
        section[data-testid="stSidebar"] .sidebar-content { background: transparent; }
        
        .app-title {
            font-size: 1.8rem;
            font-weight: 800;
            text-align: center;
            letter-spacing: -0.5px;
            background: linear-gradient(135deg, #FFFFFF 0%, #94A3B8 100%);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            margin-bottom: 0.25rem;
        }
        .app-title span {
            background: linear-gradient(135deg, #3B82F6, #8B5CF6);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }
        .app-subtitle { text-align: center; font-size: 0.7rem; font-weight: 500; color: var(--text-sub); letter-spacing: 1px; margin-bottom: 1rem; }
        
        .premium-card {
            background: var(--bg-card);
            border-radius: var(--radius-lg);
            padding: 1.25rem;
            margin-bottom: 1.25rem;
            border: 1px solid var(--border-light);
            box-shadow: var(--shadow-sm);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .premium-card:hover { transform: translateY(-2px); box-shadow: var(--shadow-md); }
        
        .section-header {
            font-size: 1rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid var(--border-light);
            color: var(--text-main);
        }
        
        .stTextInput input, .stNumberInput input, .stSelectbox select {
            background: #0F172A !important;
            border: 1px solid var(--border-light) !important;
            border-radius: 14px !important;
            padding: 12px 16px !important;
            font-size: 0.9rem !important;
            color: white !important;
            font-weight: 500 !important;
        }
        .stTextInput input:focus, .stNumberInput input:focus {
            border-color: var(--primary) !important;
            box-shadow: 0 0 0 2px rgba(59,130,246,0.2) !important;
        }
        label { font-size: 0.75rem !important; font-weight: 600 !important; color: var(--text-sub) !important; margin-bottom: 0.25rem !important; }
        
        .stButton > button {
            background: linear-gradient(135deg, var(--primary), var(--primary-dark)) !important;
            border: none !important;
            border-radius: 40px !important;
            color: white !important;
            font-weight: 700 !important;
            font-size: 1rem !important;
            padding: 0.75rem 1.5rem !important;
            width: 100% !important;
            box-shadow: 0 4px 12px rgba(59,130,246,0.3) !important;
            transition: all 0.2s ease;
        }
        .stButton > button:hover { transform: scale(1.02); box-shadow: 0 6px 16px rgba(59,130,246,0.4) !important; }
        
        .stTabs [data-baseweb="tab-list"] {
            background: var(--bg-card);
            border-radius: 40px;
            padding: 4px;
            gap: 4px;
            margin-bottom: 1.25rem;
            border: 1px solid var(--border-light);
        }
        .stTabs [data-baseweb="tab"] {
            height: 44px;
            background: transparent;
            border-radius: 32px;
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--text-sub);
            transition: all 0.2s;
        }
        .stTabs [aria-selected="true"] {
            background: var(--primary) !important;
            color: white !important;
            box-shadow: 0 2px 8px rgba(59,130,246,0.3);
        }
        
        .market-bar-container { margin-bottom: 1.2rem; }
        .market-bar-header { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px; font-size: 0.8rem; font-weight: 500; }
        .market-bar-label { color: var(--text-sub); font-weight: 600; }
        .market-bar-percentages { font-weight: 700; font-size: 0.85rem; }
        .market-bar-track { height: 10px; background: #0F172A; border-radius: 12px; overflow: hidden; display: flex; }
        .market-bar-fill { height: 100%; transition: width 0.6s cubic-bezier(0.2, 0.9, 0.4, 1.1); }
        
        .outcome-card {
            background: #0F172A;
            border-radius: 20px;
            padding: 1rem 0.5rem;
            text-align: center;
            transition: all 0.2s;
            border: 1px solid var(--border-light);
        }
        .outcome-label { font-size: 0.7rem; font-weight: 600; color: var(--text-sub); text-transform: uppercase; letter-spacing: 1px; }
        .outcome-prob { font-size: 2rem; font-weight: 800; color: white; margin: 0.25rem 0; line-height: 1; }
        .outcome-metrics { font-size: 0.65rem; display: flex; justify-content: center; gap: 0.75rem; margin-top: 0.25rem; }
        .outcome-ev, .outcome-kelly { font-weight: 600; }
        
        .mc-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            margin: 1rem 0;
        }
        @media (min-width: 640px) { .mc-grid { grid-template-columns: repeat(4, 1fr); } }
        .mc-mini-card {
            background: #0F172A;
            border-radius: 16px;
            padding: 12px 6px;
            text-align: center;
            border: 1px solid var(--border-light);
            transition: transform 0.1s;
        }
        .mc-mini-card:active { transform: scale(0.98); }
        .mc-label { font-size: 0.6rem; font-weight: 600; color: var(--text-sub); margin-bottom: 4px; white-space: nowrap; overflow-x: auto; }
        .mc-val { font-size: 1rem; font-weight: 700; color: var(--text-main); }
        
        .badge-tag {
            background: rgba(59,130,246,0.2);
            padding: 4px 10px;
            border-radius: 40px;
            font-size: 0.65rem;
            font-weight: 700;
            display: inline-block;
            color: var(--primary);
        }
        
        .backtest-card {
            background: var(--bg-card);
            border-radius: 16px;
            padding: 1rem;
            margin-bottom: 1rem;
            border: 1px solid var(--border-light);
            transition: all 0.2s;
        }
        .backtest-card:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-sm);
        }
        
        @media (max-width: 600px) {
            .stApp { padding: 0.25rem 0.5rem 1rem 0.5rem; }
            .premium-card { padding: 1rem; border-radius: 20px; }
            .outcome-prob { font-size: 1.6rem; }
            .section-header { font-size: 0.9rem; }
            .stTabs [data-baseweb="tab"] { height: 38px; font-size: 0.7rem; }
        }
        
        .stAlert { border-radius: 16px; background: rgba(59,130,246,0.1); border-left: 4px solid var(--primary); }
        hr { border-color: var(--border-light); margin: 1rem 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )