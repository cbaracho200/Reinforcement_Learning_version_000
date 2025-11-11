"""
Business-RL: Framework de Reinforcement Learning para Decisões de Negócio
"""

__version__ = "0.1.0"
__author__ = "Business-RL Team"

# Importações principais
from .core.problem import ProblemSpec, Objective, Constraint
from .core.agent import Agent, Decision
from .core.memory import UnifiedMemory
from .core.trainer import Trainer, train

# DSL
from .tools.dsl import problem, Dict, Box, Discrete, Mixed, Terms, Limit, CVaR

# Algoritmos
from .algorithms.registry import AlgorithmRegistry
from .algorithms.stable.ppo import PPOAgent
from .algorithms.stable.sac import SACAgent

# Domínios pré-construídos
from .domains.real_estate.compra_terreno import CompraTerreno
from .domains.marketing.campanha_ads import CampanhaAds

# Dashboard
from .tools.dashboard import TrainingDashboard

# Exportações principais
__all__ = [
    # Core
    'ProblemSpec', 'Agent', 'Decision', 'Trainer', 'train',
    
    # DSL
    'problem', 'Dict', 'Box', 'Discrete', 'Mixed', 'Terms', 'Limit', 'CVaR',
    
    # Algoritmos
    'PPOAgent', 'SACAgent', 'AlgorithmRegistry',
    
    # Domínios
    'CompraTerreno', 'CampanhaAds',
    
    # Tools
    'TrainingDashboard',
    
    # Meta
    '__version__'
]