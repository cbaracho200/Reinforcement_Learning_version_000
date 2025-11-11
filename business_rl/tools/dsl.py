"""
Domain Specific Language para definição de problemas.
"""

from typing import Any, Dict, List, Callable, Optional, Union, Tuple
from functools import wraps
import inspect
import numpy as np

try:
    from ..core.problem import ProblemSpec, Objective, Constraint, RiskSpec
except ImportError:
    from business_rl.core.problem import ProblemSpec, Objective, Constraint, RiskSpec


def problem(name: str = None, 
           version: str = "v0",
           description: str = None):
    """Decorador para definir problemas de negócio."""
    
    def decorator(cls):
        class ProblemWrapper(ProblemSpec):
            def __init__(self):
                self.name = name or cls.__name__
                self.version = version
                self.description = description or cls.__doc__
                self._instance = cls()
                super().__init__()
            
            def setup(self):
                if hasattr(self._instance, 'obs'):
                    self._setup_observations(self._instance.obs)
                
                if hasattr(self._instance, 'action'):
                    self._setup_actions(self._instance.action)
                
                if hasattr(self._instance, 'objectives'):
                    self._setup_objectives(self._instance.objectives)
                
                if hasattr(self._instance, 'constraints'):
                    self._setup_constraints(self._instance.constraints)
                
                if hasattr(self._instance, 'risk'):
                    self._setup_risk(self._instance.risk)
            
            def _setup_observations(self, obs_spec):
                if isinstance(obs_spec, Dict):
                    for name, spec in obs_spec.specs.items():
                        if isinstance(spec, Box):
                            self.observe(**{name: (spec.low, spec.high)})
                        elif isinstance(spec, Discrete):
                            self.observe(**{name: spec.n})
                        else:
                            self.observe(**{name: spec})
            
            def _setup_actions(self, action_spec):
                if isinstance(action_spec, Dict):
                    for name, spec in action_spec.specs.items():
                        if isinstance(spec, Box):
                            self.decide(**{name: (spec.low, spec.high)})
                        elif isinstance(spec, Discrete):
                            self.decide(**{name: spec.labels if spec.labels else spec.n})
                        else:
                            self.decide(**{name: spec})
                elif isinstance(action_spec, Mixed):
                    specs = {}
                    # Suporta tanto discrete/continuous quanto discreto/continuo
                    discrete_spec = action_spec.discrete or action_spec.discreto
                    continuous_spec = action_spec.continuous or action_spec.continuo
                    
                    if discrete_spec:
                        for name, spec in discrete_spec.specs.items():
                            if isinstance(spec, Discrete):
                                specs[name] = spec.labels if spec.labels else spec.n
                            else:
                                specs[name] = spec
                    if continuous_spec:
                        for name, spec in continuous_spec.specs.items():
                            if isinstance(spec, Box):
                                specs[name] = (spec.low, spec.high)
                            else:
                                specs[name] = spec
                    self.decide(**specs)
                elif isinstance(action_spec, Discrete):
                    self.decide(action=action_spec.labels if action_spec.labels else action_spec.n)
                elif isinstance(action_spec, Box):
                    self.decide(action=(action_spec.low, action_spec.high))
            
            def _setup_objectives(self, obj_spec):
                if isinstance(obj_spec, Terms):
                    for name, weight in obj_spec.terms.items():
                        self.maximize(**{name: Objective(name, weight)})
                elif isinstance(obj_spec, dict):
                    for name, spec in obj_spec.items():
                        if isinstance(spec, Objective):
                            self.maximize(**{name: spec})
                        else:
                            self.maximize(**{name: spec})
            
            def _setup_constraints(self, const_spec):
                if isinstance(const_spec, dict):
                    for name, spec in const_spec.items():
                        if isinstance(spec, Constraint):
                            self.constrain(**{name: spec})
                        elif isinstance(spec, Limit):
                            self.constrain(**{name: spec.to_constraint(name)})
            
            def _setup_risk(self, risk_spec):
                if isinstance(risk_spec, CVaR):
                    self.manage_risk(
                        cvar_alpha=risk_spec.alpha,
                        max_drawdown=risk_spec.max_drawdown
                    )
                elif isinstance(risk_spec, dict):
                    self.manage_risk(**risk_spec)
            
            def __getattr__(self, name):
                if name.startswith('reward_'):
                    if hasattr(self._instance, name):
                        return getattr(self._instance, name)
                raise AttributeError(f"'{self.name}' has no attribute '{name}'")
        
        ProblemWrapper.__name__ = f"{cls.__name__}Problem"
        ProblemWrapper.__qualname__ = f"{cls.__name__}Problem"
        
        return ProblemWrapper
    
    return decorator


class Dict:
    """Helper para dicionário de especificações."""
    def __init__(self, **kwargs):
        self.specs = kwargs


class Box:
    """Helper para valores contínuos."""
    def __init__(self, low: float, high: float, dtype=np.float32):
        self.low = low
        self.high = high
        self.dtype = dtype


class Discrete:
    """Helper para valores discretos."""
    def __init__(self, n: int, labels: List[str] = None):
        self.n = n
        self.labels = labels


class Mixed:
    """Helper para ações híbridas. Aceita discrete/continuous OU discreto/continuo."""
    def __init__(self, discrete=None, continuous=None, discreto=None, continuo=None):
        # Aceita ambas as formas
        self.discrete = discrete
        self.continuous = continuous
        self.discreto = discreto
        self.continuo = continuo
        
        # Para compatibilidade interna, sempre usa discrete/continuous
        if discreto is not None and discrete is None:
            self.discrete = discreto
        if continuo is not None and continuous is None:
            self.continuous = continuo


class Terms:
    """Helper para múltiplos objetivos."""
    def __init__(self, **terms):
        self.terms = terms


class Limit:
    """Helper para restrições."""
    def __init__(self, func: Callable, min_val: float = None, 
                 max_val: float = None, hard: bool = False):
        self.func = func
        self.min_val = min_val
        self.max_val = max_val
        self.hard = hard
    
    def to_constraint(self, name: str) -> Constraint:
        return Constraint(
            name=name,
            function=self.func,
            min_value=self.min_val,
            max_value=self.max_val,
            hard=self.hard
        )


class CVaR:
    """Helper para CVaR."""
    def __init__(self, alpha: float, max_drawdown: float = None):
        self.alpha = alpha
        self.max_drawdown = max_drawdown


def continuous(low: float, high: float) -> Tuple[float, float]:
    return (low, high)


def discrete(n: int) -> int:
    return n


def choices(*options) -> List:
    return list(options)
