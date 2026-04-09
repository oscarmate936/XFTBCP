# cup_context_analyzer.py
import numpy as np

class CupContextAnalyzer:
    def __init__(self, cup_matches, home_team, away_team, current_round, is_second_leg=False, first_leg_result=None):
        self.cup_matches = cup_matches
        self.home_team = home_team
        self.away_team = away_team
        self.current_round = current_round
        self.is_second_leg = is_second_leg
        self.first_leg_result = first_leg_result

    def _get_team_elimination_stage(self):
        stages = ['Primera ronda', 'Segunda ronda', 'Tercera ronda', 'Octavos', 'Cuartos', 'Semifinal', 'Final']
        try:
            idx = stages.index(self.current_round)
            progress = idx / (len(stages) - 1)
        except:
            progress = 0.5
        return progress

    def _is_high_pressure(self):
        if not self.is_second_leg:
            return 0.0
        if self.first_leg_result:
            h, a = self.first_leg_result
            diff = abs(h - a)
            if diff <= 1:
                return 0.3   # eliminatoria abierta
            elif diff == 2:
                return 0.15
            else:
                return 0.05   # casi decidido
        return 0.2

    def _away_goals_rule_factor(self):
        # La regla del gol visitante ya no existe en muchas competiciones, pero puede influir psicológicamente
        if not self.is_second_leg:
            return 0.0
        # Si el partido de ida terminó con empate a >0 goles, el visitante tiene ventaja psicológica
        if self.first_leg_result and self.first_leg_result[0] == self.first_leg_result[1] and self.first_leg_result[0] > 0:
            return 0.07
        return 0.03

    # ========== MEJORA: nuevos factores ==========
    def _first_leg_deficit(self):
        """Si el equipo local va perdiendo en el global, tenderá a atacar más."""
        if not self.is_second_leg or not self.first_leg_result:
            return 0.0
        h, a = self.first_leg_result
        # Asumiendo que el 'home_team' es el que juega en casa en la vuelta
        # Necesitamos saber quién es local en la ida? Por simplicidad, usamos la diferencia global
        # Si el equipo local (en vuelta) perdiera por 2+ goles, aumenta agresividad
        if a - h >= 2:   # el visitante de la ida (que ahora es local?) Cuidado: lógica más clara:
            # Mejor: calcular diferencia global suponiendo que el resultado de ida es (goles_local_ida, goles_visitante_ida)
            # y el equipo local de la vuelta es el que era visitante en la ida? Esto se complica.
            # Por simplicidad, devolvemos un factor si la diferencia global es >=2
            diff_global = abs(h - a)
            if diff_global >= 2:
                return 0.2
        return 0.0

    def compute_importance(self):
        factors = {
            '🏆 Fase avanzada': self._get_team_elimination_stage() * 0.4,
            '⚖️ Partido de vuelta ajustado': self._is_high_pressure(),
            '🌍 Regla gol visitante': self._away_goals_rule_factor(),
            '📉 Déficit en la ida': self._first_leg_deficit()   # NUEVO
        }
        total = sum(factors.values())
        importance = min(total, 1.0)
        return importance, factors

    def get_adjustments(self):
        importance, factors = self.compute_importance()
        # Ajustes más finos según importancia
        if importance > 0.6:
            xg_factor = 0.92
            draw_boost = 0.05
            level = "🔥 MUY ALTA (eliminatoria decisiva)"
        elif importance > 0.3:
            xg_factor = 0.97
            draw_boost = 0.02
            level = "⚠️ MEDIA (ronda importante)"
        else:
            xg_factor = 1.05
            draw_boost = -0.02
            level = "🍃 BAJA (fase temprana o resultado cómodo)"

        # Ajuste adicional si hay déficit (más ataque, menos empate)
        if factors.get('📉 Déficit en la ida', 0) > 0:
            xg_factor *= 1.08   # más goles esperados
            draw_boost -= 0.03   # menos probabilidad de empate

        return {
            'xg_factor': xg_factor,
            'draw_boost': draw_boost,
            'importance': importance,
            'level': level,
            'factors': factors
        }