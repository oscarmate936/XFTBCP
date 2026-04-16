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

    def _get_round_importance(self):
        stages = ['Primera ronda', 'Segunda ronda', 'Tercera ronda', 'Octavos', 'Cuartos', 'Semifinal', 'Final']
        try:
            idx = stages.index(self.current_round)
            return idx / (len(stages) - 1)
        except:
            return 0.5

    def _is_high_pressure(self):
        if not self.is_second_leg:
            return 0.0
        if self.first_leg_result:
            h, a = self.first_leg_result
            diff = abs(h - a)
            if diff == 0:
                return 0.35   # empate, máxima presión
            elif diff == 1:
                return 0.25
            elif diff == 2:
                return 0.15
            else:
                return 0.05
        return 0.2

    def _away_goals_rule_factor(self):
        if not self.is_second_leg:
            return 0.0
        if self.first_leg_result and self.first_leg_result[0] == self.first_leg_result[1] and self.first_leg_result[0] > 0:
            return 0.08
        return 0.02

    def _first_leg_deficit(self):
        """Factor adicional si el equipo local (en vuelta) va perdiendo en el global."""
        if not self.is_second_leg or not self.first_leg_result:
            return 0.0
        h, a = self.first_leg_result
        # Suponemos que el local de la vuelta es el visitante de la ida (cambio de campo)
        # Por simplicidad, si la diferencia global >0 a favor del visitante, el local necesita remontar.
        diff = a - h
        if diff >= 2:
            return 0.25
        elif diff == 1:
            return 0.15
        return 0.0

    def compute_importance(self):
        factors = {
            '🏆 Fase avanzada': self._get_round_importance() * 0.4,
            '⚖️ Partido de vuelta ajustado': self._is_high_pressure(),
            '🌍 Regla gol visitante': self._away_goals_rule_factor(),
            '📉 Déficit en la ida': self._first_leg_deficit()
        }
        total = sum(factors.values())
        importance = min(total, 1.0)
        return importance, factors

    def get_adjustments(self):
        importance, factors = self.compute_importance()
        # Ajustes más finos
        if importance > 0.7:
            xg_factor = 0.90
            draw_boost = 0.08
            level = "🔥 CRÍTICA (eliminatoria al límite)"
        elif importance > 0.4:
            xg_factor = 0.95
            draw_boost = 0.04
            level = "⚠️ ALTA (ronda decisiva)"
        elif importance > 0.2:
            xg_factor = 1.00
            draw_boost = 0.00
            level = "📊 MEDIA (fase intermedia)"
        else:
            xg_factor = 1.05
            draw_boost = -0.03
            level = "🍃 BAJA (fase temprana)"

        if factors.get('📉 Déficit en la ida', 0) > 0:
            xg_factor *= 1.10   # más ataque
            draw_boost -= 0.05   # menos empates

        return {
            'xg_factor': xg_factor,
            'draw_boost': draw_boost,
            'importance': importance,
            'level': level,
            'factors': factors
        }
