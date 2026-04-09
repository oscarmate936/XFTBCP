# train_cerebro_cups.py
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import time
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Instalación automática de dependencias si es necesario
try:
    import optuna
    from xgboost import XGBClassifier
    from sklearn.ensemble import RandomForestClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import log_loss
except ImportError:
    print("📦 Instalando dependencias...")
    os.system(f"{sys.executable} -m pip install optuna xgboost scikit-learn pandas requests joblib -q")
    import optuna
    from xgboost import XGBClassifier
    from sklearn.ensemble import RandomForestClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import log_loss

API_KEY = "406157"
BASE_URL = f"https://www.thesportsdb.com/api/v1/json/{API_KEY}/"

# Lista de copas a considerar (ID, nombre, formato de temporada)
COPAS_CONFIRMADAS = [
    (4480, "Champions League", "split"),
    (4481, "Europa League", "split"),
    (5071, "Conference League", "split"),
    (4483, "Copa del Rey", "split"),
    (4490, "FA Cup", "split"),
    (4485, "Coppa Italia", "split"),
    (4486, "DFB-Pokal", "split"),
    (4487, "Coupe de France", "split"),
    (4488, "Mundial de Clubes", "year"),
]

TEMPORADAS_SPLIT = ["2024-2025", "2025-2026"]
TEMPORADAS_YEAR = ["2024", "2025", "2026"]

def get_cup_events(cup_id, season):
    url = f"{BASE_URL}eventsseason.php?id={cup_id}&s={season}"
    try:
        r = requests.get(url, timeout=15)
        data = r.json()
        return data.get('events', [])
    except Exception as e:
        print(f"  Error al obtener {cup_id} {season}: {e}")
        return []

def calcular_stats_avanzadas(goles, fechas, prom_liga, C=10.0, hl=30):
    """Misma función que en data_processing.py"""
    if not goles:
        return prom_liga, 0.0, 0.0
    hoy = fechas[-1]
    pesos = np.array([np.exp(-(np.log(2)/hl) * max(0, (hoy - f).days)) for f in fechas])
    vals = np.array(goles)
    prom_pond = np.sum(vals * pesos) / np.sum(pesos)
    xg_bayes = ((prom_pond * np.sum(pesos)) + (prom_liga * C)) / (np.sum(pesos) + C)
    volatilidad = np.std(goles) if len(goles) > 1 else 0.0
    momentum = np.polyfit(np.arange(len(goles)), goles, 1)[0] if len(goles) >= 3 else 0.0
    return xg_bayes, volatilidad, momentum

def resolver_colley(n, partidos, eq_idx):
    M = np.diag([2.0] * n)
    b = np.ones(n)
    for h, v, r in partidos:
        M[h,h] += 1
        M[v,v] += 1
        M[h,v] -= 1
        M[v,h] -= 1
        if r == 0:
            b[h] += 0.5
            b[v] -= 0.5
        elif r == 2:
            b[h] -= 0.5
            b[v] += 0.5
    try:
        sol = np.linalg.solve(M, b)
        return {eq: r for eq, r in zip(eq_idx.keys(), sol)}
    except:
        return {eq: 0.5 for eq in eq_idx}

print("🚀 Iniciando recolección de datos para CEREBRO DE COPAS...")
data_final = []

for idx, (cup_id, cup_nombre, formato) in enumerate(COPAS_CONFIRMADAS, 1):
    print(f"\n[{idx}/{len(COPAS_CONFIRMADAS)}] {cup_nombre}")
    temporadas = TEMPORADAS_SPLIT if formato == "split" else TEMPORADAS_YEAR
    for season in temporadas:
        print(f"  📅 Temporada {season}...")
        events = get_cup_events(cup_id, season)
        if not events:
            print(f"    ⚠️ No hay eventos para {cup_nombre} en {season}")
            continue
        events = [e for e in events if e.get('dateEvent') and e.get('intHomeScore') is not None]
        events.sort(key=lambda x: x['dateEvent'])
        if not events:
            continue
        total_goles = sum(int(e['intHomeScore']) + int(e['intAwayScore']) for e in events)
        prom_liga_temp = total_goles / len(events) if events else 2.5
        print(f"    Promedio goles: {prom_liga_temp:.2f}")
        
        est = {}
        hist_c = []
        eq_names = list(set([e['strHomeTeam'] for e in events] + [e['strAwayTeam'] for e in events]))
        idx_m = {name: i for i, name in enumerate(eq_names)}
        partidos_procesados = 0
        
        for ev in events:
            hl, vl = ev['strHomeTeam'], ev['strAwayTeam']
            gh, ga = int(ev['intHomeScore']), int(ev['intAwayScore'])
            f = datetime.strptime(ev['dateEvent'], '%Y-%m-%d')
            for q in (hl, vl):
                if q not in est:
                    est[q] = {'gf': [], 'gc': [], 'f': [], 'elo': 1500, 'pj': 0, 'pts_esp': []}
            if est[hl]['pj'] >= 4 and est[vl]['pj'] >= 4:
                xg_l, vol_l, mom_l = calcular_stats_avanzadas(est[hl]['gf'], est[hl]['f'], prom_liga_temp/2)
                xg_v, vol_v, mom_v = calcular_stats_avanzadas(est[vl]['gf'], est[vl]['f'], prom_liga_temp/2)
                def_l, _, _ = calcular_stats_avanzadas(est[hl]['gc'], est[hl]['f'], prom_liga_temp/2)
                def_v, _, _ = calcular_stats_avanzadas(est[vl]['gc'], est[vl]['f'], prom_liga_temp/2)
                c_ratings = resolver_colley(len(idx_m), hist_c, idx_m)
                
                gf_h = est[hl]['gf']
                gc_h = est[hl]['gc']
                gf_a = est[vl]['gf']
                gc_a = est[vl]['gc']
                
                gd_h_3 = sum(gf_h[-3:]) - sum(gc_h[-3:]) if len(gf_h) >= 3 else 0
                gd_a_3 = sum(gf_a[-3:]) - sum(gc_a[-3:]) if len(gf_a) >= 3 else 0
                gd_h_5 = sum(gf_h[-5:]) - sum(gc_h[-5:]) if len(gf_h) >= 5 else 0
                gd_a_5 = sum(gf_a[-5:]) - sum(gc_a[-5:]) if len(gf_a) >= 5 else 0
                gd_h_10 = sum(gf_h[-10:]) - sum(gc_h[-10:]) if len(gf_h) >= 10 else 0
                gd_a_10 = sum(gf_a[-10:]) - sum(gc_a[-10:]) if len(gf_a) >= 10 else 0
                
                avg_gf_h_3 = np.mean(gf_h[-3:]) if len(gf_h) >= 3 else prom_liga_temp/2
                avg_gc_h_3 = np.mean(gc_h[-3:]) if len(gc_h) >= 3 else prom_liga_temp/2
                avg_gf_a_3 = np.mean(gf_a[-3:]) if len(gf_a) >= 3 else prom_liga_temp/2
                avg_gc_a_3 = np.mean(gc_a[-3:]) if len(gc_a) >= 3 else prom_liga_temp/2
                
                def btts_ratio(gf, gc, n=5):
                    if len(gf) < n: return 0.5
                    cnt = 0
                    for i in range(-n, 0):
                        if gf[i] > 0 and gc[i] > 0:
                            cnt += 1
                    return cnt / n
                btts_h_5 = btts_ratio(gf_h, gc_h)
                btts_a_5 = btts_ratio(gf_a, gc_a)
                
                pts_h = est[hl]['pts_esp']
                pts_a = est[vl]['pts_esp']
                win_streak_h = 0
                for p in reversed(pts_h):
                    if p == 3: win_streak_h += 1
                    else: break
                loss_streak_h = 0
                for p in reversed(pts_h):
                    if p == 0: loss_streak_h += 1
                    else: break
                win_streak_a = 0
                for p in reversed(pts_a):
                    if p == 3: win_streak_a += 1
                    else: break
                loss_streak_a = 0
                for p in reversed(pts_a):
                    if p == 0: loss_streak_a += 1
                    else: break
                
                pesos_racha = [0.1, 0.15, 0.2, 0.25, 0.3]
                r_l = sum(p*w for p,w in zip(pts_h[-5:], pesos_racha)) if len(pts_h) >= 5 else 1.5
                r_v = sum(p*w for p,w in zip(pts_a[-5:], pesos_racha)) if len(pts_a) >= 5 else 1.5
                
                data_final.append({
                    'ventaja_local': xg_l * (c_ratings.get(hl, 0.5) + 0.5),
                    'ventaja_visita': xg_v * (c_ratings.get(vl, 0.5) + 0.5),
                    'defensa_local': def_l,
                    'defensa_visita': def_v,
                    'elo_diff': est[hl]['elo'] - est[vl]['elo'],
                    'racha_esp_local': r_l,
                    'racha_esp_visita': r_v,
                    'volatilidad_local': vol_l,
                    'volatilidad_visita': vol_v,
                    'momentum_local': mom_l,
                    'momentum_visita': mom_v,
                    'gd_h_3': gd_h_3, 'gd_a_3': gd_a_3,
                    'gd_h_5': gd_h_5, 'gd_a_5': gd_a_5,
                    'gd_h_10': gd_h_10, 'gd_a_10': gd_a_10,
                    'avg_gf_h_3': avg_gf_h_3,
                    'avg_gc_h_3': avg_gc_h_3,
                    'avg_gf_a_3': avg_gf_a_3,
                    'avg_gc_a_3': avg_gc_a_3,
                    'btts_h_5': btts_h_5,
                    'btts_a_5': btts_a_5,
                    'win_streak_h': win_streak_h,
                    'loss_streak_h': loss_streak_h,
                    'win_streak_a': win_streak_a,
                    'loss_streak_a': loss_streak_a,
                    'resultado': 0 if gh > ga else (1 if gh == ga else 2),
                    'fecha': f
                })
                partidos_procesados += 1
            
            # Actualizar estadísticas
            est[hl]['gf'].append(gh)
            est[hl]['gc'].append(ga)
            est[hl]['f'].append(f)
            est[hl]['pj'] += 1
            est[vl]['gf'].append(ga)
            est[vl]['gc'].append(gh)
            est[vl]['f'].append(f)
            est[vl]['pj'] += 1
            pts_h_local = 3 if gh > ga else (1 if gh == ga else 0)
            pts_v_local = 3 if ga > gh else (1 if gh == ga else 0)
            est[hl]['pts_esp'].append(pts_h_local)
            est[vl]['pts_esp'].append(pts_v_local)
            hist_c.append((idx_m[hl], idx_m[vl], 0 if gh > ga else (1 if gh == ga else 2)))
            exp = 1 / (1 + 10**((est[vl]['elo'] - est[hl]['elo']) / 400))
            resultado = 1 if gh > ga else (0.5 if gh == ga else 0)
            est[hl]['elo'] += 20 * (resultado - exp)
            est[vl]['elo'] += 20 * ((1 - resultado) - (1 - exp))
        
        print(f"    Partidos útiles generados: {partidos_procesados}")
        time.sleep(0.3)

df = pd.DataFrame(data_final)
if df.empty:
    print("\n❌ No se recopilaron suficientes datos. Abortando.")
    sys.exit(1)

print(f"\n📊 Total de partidos recopilados para copas: {len(df)}")
df = df.sort_values('fecha')
X = df.drop(columns=['resultado', 'fecha'])
y = df['resultado']

print("\n🔍 Optimización con Optuna y validación temporal (TimeSeriesSplit)...")

def objective(trial):
    params = {
        'xgb__n_estimators': trial.suggest_int('xgb_n_estimators', 50, 150),
        'xgb__max_depth': trial.suggest_int('xgb_max_depth', 3, 7),
        'xgb__learning_rate': trial.suggest_float('xgb_lr', 0.01, 0.1, log=True),
        'rf__n_estimators': trial.suggest_int('rf_n_estimators', 80, 200),
        'rf__max_depth': trial.suggest_int('rf_max_depth', 5, 12),
        'rf__min_samples_split': trial.suggest_int('rf_min_split', 2, 10)
    }
    xgb_clf = XGBClassifier(
        n_estimators=params['xgb__n_estimators'],
        max_depth=params['xgb__max_depth'],
        learning_rate=params['xgb__learning_rate'],
        subsample=0.8, verbosity=0, random_state=42
    )
    rf_clf = RandomForestClassifier(
        n_estimators=params['rf__n_estimators'],
        max_depth=params['rf__max_depth'],
        min_samples_split=params['rf__min_samples_split'],
        random_state=42
    )
    lr_pipe = Pipeline([('s', StandardScaler()), ('l', LogisticRegression(max_iter=500))])
    stack = StackingClassifier(
        estimators=[('xgb', xgb_clf), ('rf', rf_clf), ('lr', lr_pipe)],
        final_estimator=LogisticRegression(),
        cv=3
    )
    tscv = TimeSeriesSplit(n_splits=4)
    scores = []
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        stack.fit(X_train, y_train)
        probas = stack.predict_proba(X_val)
        scores.append(log_loss(y_val, probas))
    return np.mean(scores)

study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=30, show_progress_bar=True)
best_params = study.best_params
print(f"Mejores hiperparámetros: {best_params}")

print("\n🧠 Entrenando modelo final con todos los datos...")
xgb_final = XGBClassifier(
    n_estimators=best_params['xgb_n_estimators'],
    max_depth=best_params['xgb_max_depth'],
    learning_rate=best_params['xgb_lr'],
    subsample=0.8, verbosity=0, random_state=42
)
rf_final = RandomForestClassifier(
    n_estimators=best_params['rf_n_estimators'],
    max_depth=best_params['rf_max_depth'],
    min_samples_split=best_params['rf_min_split'],
    random_state=42
)
lr_pipe = Pipeline([('s', StandardScaler()), ('l', LogisticRegression(max_iter=500))])
stack_final = StackingClassifier(
    estimators=[('xgb', xgb_final), ('rf', rf_final), ('lr', lr_pipe)],
    final_estimator=LogisticRegression(),
    cv=5
)
stack_final.fit(X, y)

print("🧠 Calibrando probabilidades con isotonic regression...")
calibrated = CalibratedClassifierCV(stack_final, method='isotonic', cv=5)
calibrated.fit(X, y)

cerebro_cup = {
    'modelo_clasificacion': calibrated,
    'feature_names': list(X.columns),
    'n_matches': len(df),
    'timestamp': datetime.now().isoformat(),
    'best_params': best_params,
    'tipo': 'copas'
}
filename = 'quantum_cerebro_cup_final.pkl'
joblib.dump(cerebro_cup, filename, compress=3)
size = os.path.getsize(filename) / 1e6
print(f"\n✅ CEREBRO PARA COPAS GUARDADO en '{filename}'. Tamaño: {size:.2f} MB")
print("🎉 Puedes usar este cerebro en app_cup.py colocándolo en la raíz del proyecto.") 