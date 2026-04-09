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
                return 0.3
            return 0.1
        return 0.2

    def _away_goals_rule_factor(self):
        if not self.is_second_leg:
            return 0.0
        return 0.05

    def compute_importance(self):
        factors = {
            '🏆 Fase avanzada': self._get_team_elimination_stage() * 0.4,
            '⚖️ Partido de vuelta ajustado': self._is_high_pressure(),
            '🌍 Regla gol visitante': self._away_goals_rule_factor()
        }
        total = sum(factors.values())
        importance = min(total, 1.0)
        return importance, factors

    def get_adjustments(self):
        importance, factors = self.compute_importance()
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
        return {
            'xg_factor': xg_factor,
            'draw_boost': draw_boost,
            'importance': importance,
            'level': level,
            'factors': factors
        }