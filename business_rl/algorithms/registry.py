"""
Registro e seleção automática de algoritmos.
"""

from typing import Dict, Type, Optional
from ..core.agent import Agent
from ..core.problem import ProblemSpec


class AlgorithmRegistry:
    """Registro central de algoritmos."""
    
    _algorithms: Dict[str, Type[Agent]] = {}
    
    @classmethod
    def register(cls, name: str, agent_class: Type[Agent]):
        """Registra um novo algoritmo."""
        cls._algorithms[name.lower()] = agent_class
    
    @classmethod
    def get(cls, name: str) -> Type[Agent]:
        """Obtém classe do algoritmo."""
        name = name.lower()
        
        if name not in cls._algorithms:
            raise ValueError(f"Algoritmo '{name}' não encontrado. "
                           f"Disponíveis: {list(cls._algorithms.keys())}")
        
        return cls._algorithms[name]
    
    @classmethod
    def list_available(cls) -> list:
        """Lista algoritmos disponíveis."""
        return list(cls._algorithms.keys())
    
    @classmethod
    def auto_select(cls, problem: ProblemSpec) -> str:
        """Seleciona algoritmo automaticamente baseado no problema."""
        
        info = problem.get_info()
        
        # Analisa características do problema
        has_continuous = info['has_continuous_actions']
        has_discrete = info['has_discrete_actions']
        has_constraints = info['n_constraints'] > 0
        has_multi_obj = info['n_objectives'] > 1
        is_risk_managed = info['risk_managed']
        
        # Regras de seleção
        if has_constraints:
            if has_continuous and not has_discrete:
                return 'sac-lagrangian'
            else:
                return 'ppo-lagrangian'
        
        elif has_multi_obj:
            return 'mo-ppo'
        
        elif is_risk_managed:
            if has_continuous and not has_discrete:
                return 'risk-sac'
            else:
                return 'risk-ppo'
        
        elif has_continuous and has_discrete:
            return 'ppo'  # PPO funciona bem com híbrido
        
        elif has_continuous:
            return 'sac'
        
        else:
            return 'ppo'
    
    @classmethod
    def get_config_template(cls, algorithm: str) -> Dict:
        """Retorna template de configuração para o algoritmo."""
        
        templates = {
            'ppo': {
                'learning_rate': 3e-4,
                'n_epochs': 10,
                'clip_range': 0.2,
                'ent_coef': 0.01,
                'vf_coef': 0.5,
                'batch_size': 64,
                'gamma': 0.99
            },
            'sac': {
                'learning_rate': 3e-4,
                'batch_size': 256,
                'tau': 0.005,
                'gamma': 0.99,
                'alpha': 0.2,
                'auto_entropy': True
            },
            'ppo-lagrangian': {
                'learning_rate': 3e-4,
                'n_epochs': 10,
                'clip_range': 0.2,
                'constraint_threshold': 0.1,
                'lagrange_lr': 1e-3,
                'batch_size': 64,
                'gamma': 0.99
            }
        }
        
        algorithm = algorithm.lower()
        return templates.get(algorithm, {})


# Importa e registra algoritmos implementados
def register_all_algorithms():
    """Registra todos os algoritmos disponíveis."""
    
    # Importações locais para evitar circular imports
    try:
        from .stable.ppo import PPOAgent, PPOWithTrustRegion
    except ImportError:
        print("Aviso: Não foi possível importar PPOAgent")
        PPOAgent = None
        PPOWithTrustRegion = None
    
    try:
        from .stable.sac import SACAgent
    except ImportError:
        print("Aviso: Não foi possível importar SACAgent")
        SACAgent = None
    
    # Registra algoritmos estáveis
    AlgorithmRegistry.register('ppo', PPOAgent)
    AlgorithmRegistry.register('ppo-tr', PPOWithTrustRegion)
    AlgorithmRegistry.register('sac', SACAgent)
    
    # TODO: Registrar algoritmos experimentais quando implementados
    # from .experimental.rcpo import RCPOAgent
    # from .experimental.mo_ppo import MOPPOAgent
    # AlgorithmRegistry.register('rcpo', RCPOAgent)
    # AlgorithmRegistry.register('mo-ppo', MOPPOAgent)


# Registra algoritmos ao importar o módulo
register_all_algorithms()