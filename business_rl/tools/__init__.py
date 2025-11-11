"""Ferramentas e utilitários."""

try:
    from business_rl.tools.dsl import (
        problem, Dict, Box, Discrete, Mixed, 
        Terms, Limit, CVaR, continuous, discrete, choices
    )
    from business_rl.tools.dashboard import TrainingDashboard

    __all__ = [
        'problem', 'Dict', 'Box', 'Discrete', 'Mixed',
        'Terms', 'Limit', 'CVaR', 'continuous', 'discrete', 'choices',
        'TrainingDashboard'
    ]
except ImportError:
    __all__ = []
