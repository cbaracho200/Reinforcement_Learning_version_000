"""
Interface unificada para agentes de RL.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass


@dataclass
class Decision:
    """Resultado de uma decisão do agente."""
    action: Any
    value: Optional[float] = None
    confidence: Optional[float] = None
    log_prob: Optional[float] = None
    entropy: Optional[float] = None
    info: Dict[str, Any] = None
    
    def to_dict(self) -> Dict:
        """Converte para dicionário."""
        return {
            'action': self.action,
            'value': self.value,
            'confidence': self.confidence,
            'info': self.info or {}
        }


class Agent(ABC):
    """Classe base para todos os agentes."""
    
    def __init__(self, problem_spec, config=None):
        self.problem_spec = problem_spec
        self.config = config or {}
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Redes neurais
        self.policy_net = None
        self.value_net = None
        self.target_nets = {}
        
        # Estado interno
        self.training = True
        self.steps = 0
        self.episodes = 0
        
        # Métricas
        self.metrics = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'kl_divergence': [],
            'constraint_violations': []
        }
        
        # Inicializa redes
        self._build_networks()
    
    @abstractmethod
    def _build_networks(self):
        """Constrói as redes neurais do agente."""
        pass
    
    @abstractmethod
    def act(self, observation: np.ndarray, 
            deterministic: bool = False) -> Decision:
        """Escolhe uma ação dado uma observação."""
        pass
    
    @abstractmethod
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Atualiza o agente com um batch de experiências."""
        pass
    
    def save(self, path: str):
        """Salva o modelo."""
        checkpoint = {
            'policy_net': self.policy_net.state_dict(),
            'value_net': self.value_net.state_dict() if self.value_net else None,
            'config': self.config,
            'steps': self.steps,
            'episodes': self.episodes,
            'metrics': self.metrics
        }
        
        # Adiciona target networks se existirem
        for name, net in self.target_nets.items():
            checkpoint[f'target_{name}'] = net.state_dict()
        
        torch.save(checkpoint, path)
    
    def load(self, path: str):
        """Carrega o modelo."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        if self.value_net and checkpoint.get('value_net'):
            self.value_net.load_state_dict(checkpoint['value_net'])
        
        # Carrega target networks
        for name, net in self.target_nets.items():
            key = f'target_{name}'
            if key in checkpoint:
                net.load_state_dict(checkpoint[key])
        
        self.steps = checkpoint.get('steps', 0)
        self.episodes = checkpoint.get('episodes', 0)
        self.metrics = checkpoint.get('metrics', self.metrics)
    
    def train_mode(self, mode: bool = True):
        """Define modo de treinamento."""
        self.training = mode
        if self.policy_net:
            self.policy_net.train(mode)
        if self.value_net:
            self.value_net.train(mode)
    
    def eval_mode(self):
        """Define modo de avaliação."""
        self.train_mode(False)
    
    def to(self, device):
        """Move redes para dispositivo."""
        self.device = device
        if self.policy_net:
            self.policy_net.to(device)
        if self.value_net:
            self.value_net.to(device)
        for net in self.target_nets.values():
            net.to(device)
        return self
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retorna métricas recentes."""
        metrics = {}
        for key, values in self.metrics.items():
            if values:
                # Retorna média das últimas 100 medições
                recent = values[-100:]
                metrics[key] = {
                    'mean': np.mean(recent),
                    'std': np.std(recent),
                    'min': np.min(recent),
                    'max': np.max(recent)
                }
        return metrics
    
    def reset_metrics(self):
        """Limpa métricas acumuladas."""
        for key in self.metrics:
            self.metrics[key] = []


class PolicyNetwork(nn.Module):
    """Rede base para políticas."""
    
    def __init__(self, obs_dim: int, act_dim: int, 
                 hidden_dims: list = [256, 256],
                 activation: str = 'relu'):
        super().__init__()
        
        # Escolhe função de ativação
        act_fn = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'elu': nn.ELU,
            'swish': nn.SiLU
        }[activation]
        
        # Constrói MLP
        layers = []
        prev_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                act_fn(),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        self.trunk = nn.Sequential(*layers)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = prev_dim
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass base."""
        return self.trunk(obs)


class ValueNetwork(nn.Module):
    """Rede base para funções valor."""
    
    def __init__(self, obs_dim: int, 
                 hidden_dims: list = [256, 256],
                 activation: str = 'relu'):
        super().__init__()
        
        # Escolhe função de ativação
        act_fn = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'elu': nn.ELU,
            'swish': nn.SiLU
        }[activation]
        
        # Constrói MLP
        layers = []
        prev_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                act_fn(),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        # Camada de saída
        layers.append(nn.Linear(prev_dim, 1))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Calcula valor do estado."""
        return self.net(obs).squeeze(-1)