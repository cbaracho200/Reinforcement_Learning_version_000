"""
Business-RL: Framework de Reinforcement Learning para Decisões de Negócio
"""

__version__ = "0.1.0"
__author__ = "Business-RL Team"

# Importações principais - usando imports absolutos
try:
    from business_rl.core.problem import ProblemSpec, Objective, Constraint
    from business_rl.core.agent import Agent, Decision
    from business_rl.core.memory import UnifiedMemory
    from business_rl.core.trainer import Trainer, train

    # DSL
    from business_rl.tools.dsl import problem, Dict, Box, Discrete, Mixed, Terms, Limit, CVaR

    # Algoritmos
    from business_rl.algorithms.registry import AlgorithmRegistry
    from business_rl.algorithms.stable.ppo import PPOAgent
    from business_rl.algorithms.stable.sac import SACAgent

    # Dashboard
    from business_rl.tools.dashboard import TrainingDashboard

    # Exportações principais
    __all__ = [
        # Core
        'ProblemSpec', 'Agent', 'Decision', 'Trainer', 'train',
        
        # DSL
        'problem', 'Dict', 'Box', 'Discrete', 'Mixed', 'Terms', 'Limit', 'CVaR',
        
        # Algoritmos
        'PPOAgent', 'SACAgent', 'AlgorithmRegistry',
        
        # Tools
        'TrainingDashboard',
        
        # Meta
        '__version__'
    ]
    
    print("Business-RL carregado com sucesso!")
    
except ImportError as e:
    print(f"Aviso: Alguns módulos não puderam ser importados: {e}")
    __all__ = []
