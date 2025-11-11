"""
Definição da especificação de problemas de negócio.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union, Callable
from abc import ABC, abstractmethod
import numpy as np
from gymnasium import spaces


@dataclass
class Objective:
    """Define um objetivo a ser maximizado."""
    name: str
    weight: float = 1.0
    normalize: bool = True
    target: Optional[float] = None
    
    def compute(self, value: float, baseline: float = 0.0) -> float:
        """Calcula o valor normalizado do objetivo."""
        if self.normalize and self.target:
            return (value - baseline) / (self.target - baseline)
        return value


@dataclass
class Constraint:
    """Define uma restrição no problema."""
    name: str
    function: Callable
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    penalty_weight: float = 1.0
    hard: bool = False  # Se True, para o episódio quando violada
    
    def is_satisfied(self, state: Dict[str, Any]) -> bool:
        """Verifica se a restrição está satisfeita."""
        value = self.function(state)
        if self.min_value is not None and value < self.min_value:
            return False
        if self.max_value is not None and value > self.max_value:
            return False
        return True
    
    def violation(self, state: Dict[str, Any]) -> float:
        """Calcula o grau de violação da restrição."""
        value = self.function(state)
        violation = 0.0
        if self.min_value is not None:
            violation += max(0, self.min_value - value)
        if self.max_value is not None:
            violation += max(0, value - self.max_value)
        return violation * self.penalty_weight


@dataclass
class RiskSpec:
    """Especificação de gerenciamento de risco."""
    cvar_alpha: float = 0.1
    max_drawdown: float = 0.3
    var_confidence: float = 0.95
    robust_radius: float = 0.1  # Para otimização robusta
    

class ProblemSpec(ABC):
    """Classe base para especificação de problemas de negócio."""
    
    def __init__(self):
        self.observation_spec = {}
        self.action_spec = {}
        self.objectives = []
        self.constraints = []
        self.risk_spec = None
        self.metadata = {}
        
        # Chama o método de configuração do usuário
        self.setup()
        
        # Constrói os espaços do Gymnasium
        self._build_spaces()
    
    @abstractmethod
    def setup(self):
        """Define o problema. Deve ser implementado pelo usuário."""
        pass
    
    def observe(self, **kwargs):
        """Define as observações do ambiente."""
        for name, spec in kwargs.items():
            if isinstance(spec, tuple) and len(spec) == 2:
                # Box contínuo: (min, max)
                self.observation_spec[name] = {
                    'type': 'continuous',
                    'low': spec[0],
                    'high': spec[1],
                    'dtype': np.float32
                }
            elif isinstance(spec, int):
                # Discreto
                self.observation_spec[name] = {
                    'type': 'discrete',
                    'n': spec,
                    'dtype': np.int32
                }
            elif isinstance(spec, dict):
                # Especificação completa
                self.observation_spec[name] = spec
    
    def decide(self, **kwargs):
        """Define as ações disponíveis."""
        for name, spec in kwargs.items():
            if isinstance(spec, list):
                # Ação discreta com labels
                self.action_spec[name] = {
                    'type': 'discrete',
                    'choices': spec,
                    'n': len(spec)
                }
            elif isinstance(spec, tuple) and len(spec) == 2:
                # Ação contínua: (min, max)
                self.action_spec[name] = {
                    'type': 'continuous',
                    'low': spec[0],
                    'high': spec[1],
                    'dtype': np.float32
                }
            elif isinstance(spec, dict):
                self.action_spec[name] = spec
    
    def maximize(self, **objectives):
        """Define os objetivos a maximizar."""
        for name, obj in objectives.items():
            if isinstance(obj, Objective):
                obj.name = name
                self.objectives.append(obj)
            elif isinstance(obj, (int, float)):
                self.objectives.append(Objective(name, weight=obj))
    
    def constrain(self, **constraints):
        """Define as restrições do problema."""
        for name, const in constraints.items():
            if isinstance(const, Constraint):
                const.name = name
                self.constraints.append(const)
    
    def manage_risk(self, **risk_params):
        """Define parâmetros de gerenciamento de risco."""
        self.risk_spec = RiskSpec(**risk_params)
    
    def _build_spaces(self):
        """Constrói os espaços do Gymnasium."""
        # Espaço de observação
        obs_spaces = {}
        for name, spec in self.observation_spec.items():
            if spec['type'] == 'continuous':
                obs_spaces[name] = spaces.Box(
                    low=spec['low'], 
                    high=spec['high'],
                    dtype=spec.get('dtype', np.float32)
                )
            elif spec['type'] == 'discrete':
                obs_spaces[name] = spaces.Discrete(spec['n'])
        
        self.observation_space = spaces.Dict(obs_spaces)
        
        # Espaço de ação
        act_spaces = {}
        for name, spec in self.action_spec.items():
            if spec['type'] == 'continuous':
                act_spaces[name] = spaces.Box(
                    low=spec['low'],
                    high=spec['high'],
                    dtype=spec.get('dtype', np.float32)
                )
            elif spec['type'] == 'discrete':
                act_spaces[name] = spaces.Discrete(spec['n'])
        
        if len(act_spaces) == 0:
            self.action_space = spaces.Discrete(1)  # Ação dummy se não houver ações definidas
        elif len(act_spaces) == 1:
            self.action_space = list(act_spaces.values())[0]
        else:
            self.action_space = spaces.Dict(act_spaces)
    
    def compute_reward(self, state: Dict, action: Dict, next_state: Dict) -> Dict[str, float]:
        """Calcula as recompensas para cada objetivo."""
        rewards = {}
        
        # Calcula recompensa para cada objetivo
        for obj in self.objectives:
            if hasattr(self, f'reward_{obj.name}'):
                reward_fn = getattr(self, f'reward_{obj.name}')
                rewards[obj.name] = reward_fn(state, action, next_state)
            else:
                rewards[obj.name] = 0.0
        
        # Penaliza violações de restrições
        total_violation = 0.0
        for const in self.constraints:
            total_violation += const.violation(next_state)
        
        rewards['constraint_penalty'] = -total_violation
        
        return rewards
    
    def aggregate_rewards(self, rewards: Dict[str, float]) -> float:
        """Agrega múltiplos objetivos em uma recompensa escalar."""
        total = 0.0
        
        # Soma ponderada dos objetivos
        for obj in self.objectives:
            if obj.name in rewards:
                total += obj.weight * rewards[obj.name]
        
        # Adiciona penalidades
        total += rewards.get('constraint_penalty', 0.0)
        
        return total
    
    def get_info(self) -> Dict[str, Any]:
        """Retorna informações sobre o problema."""
        return {
            'name': getattr(self, 'name', self.__class__.__name__),
            'n_objectives': len(self.objectives),
            'n_constraints': len(self.constraints),
            'has_continuous_actions': any(
                s['type'] == 'continuous' 
                for s in self.action_spec.values()
            ),
            'has_discrete_actions': any(
                s['type'] == 'discrete'
                for s in self.action_spec.values()
            ),
            'risk_managed': self.risk_spec is not None,
            'observation_dim': self._get_obs_dim(),
            'action_dim': self._get_act_dim(),
            'observation_space': self.observation_spec,
            'action_space': self.action_spec,
            'action_type': 'hybrid' if (any(s['type'] == 'continuous' for s in self.action_spec.values()) and 
                                       any(s['type'] == 'discrete' for s in self.action_spec.values())) 
                          else ('continuous' if any(s['type'] == 'continuous' for s in self.action_spec.values()) 
                          else 'discrete')
        }
    
    def _get_obs_dim(self) -> int:
        """Calcula dimensão do espaço de observação."""
        dim = 0
        for spec in self.observation_spec.values():
            if spec['type'] == 'continuous':
                dim += 1  # Box sempre conta como 1
            else:  # discrete
                dim += 1  # Discrete também conta como 1 (não one-hot)
        return dim
    
    def _get_act_dim(self) -> int:
        """Calcula dimensão do espaço de ação."""
        dim = 0
        for spec in self.action_spec.values():
            if spec['type'] == 'continuous':
                dim += 1  # Box sempre conta como 1
            else:  # discrete
                dim += 1  # Discrete também conta como 1
        return dim if dim > 0 else 1  # Retorna 1 se não houver ações
    
