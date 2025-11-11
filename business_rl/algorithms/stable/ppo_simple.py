"""
Implementação simplificada do PPO para teste.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import numpy as np
from typing import Dict, Any

from ...core.agent import Agent, PolicyNetwork, ValueNetwork, Decision


class PPOAgent(Agent):
    """PPO simplificado."""
    
    def __init__(self, problem_spec, config=None):
        default_config = {
            'learning_rate': 3e-4,
            'n_epochs': 10,
            'clip_range': 0.2,
            'ent_coef': 0.01,
            'vf_coef': 0.5,
            'gamma': 0.99
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(problem_spec, default_config)
        
    def _build_networks(self):
        """Constrói redes do PPO."""
        
        # Calcula dimensões reais
        obs_dim = self._calculate_obs_dim()
        act_dim = self._calculate_act_dim()
        
        # Política simples
        self.policy_net = SimplePolicy(obs_dim, act_dim).to(self.device)
        self.value_net = ValueNetwork(obs_dim, [256, 256]).to(self.device)
        
        # Otimizador
        self.optimizer = torch.optim.Adam(
            list(self.policy_net.parameters()) + 
            list(self.value_net.parameters()),
            lr=self.config['learning_rate']
        )
    
    def _calculate_obs_dim(self):
        """Calcula dimensão real das observações."""
        dim = 0
        for spec in self.problem_spec.observation_spec.values():
            dim += 1  # Cada observação conta como 1
        return dim
    
    def _calculate_act_dim(self):
        """Calcula dimensão real das ações."""
        # Por simplicidade, assume ações discretas
        for spec in self.problem_spec.action_spec.values():
            if spec['type'] == 'discrete':
                return spec.get('n', 4)
        return 4  # Default
    
    def act(self, observation: np.ndarray, deterministic: bool = False) -> Decision:
        """Seleciona ação."""
        
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Política
            action_probs = self.policy_net(obs_tensor)
            dist = Categorical(action_probs)
            
            if deterministic:
                action = dist.probs.argmax(dim=-1)
            else:
                action = dist.sample()
            
            log_prob = dist.log_prob(action).sum()
            entropy = dist.entropy().sum()
            
            # Valor
            value = self.value_net(obs_tensor)
            
        return Decision(
            action=action.cpu().numpy()[0] if hasattr(action, 'cpu') else action,
            value=value.item(),
            log_prob=log_prob.item(),
            entropy=entropy.item()
        )
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Atualização simplificada."""
        return {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0
        }


class SimplePolicy(nn.Module):
    """Política simplificada."""
    
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, act_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)
        return x


# Alias para compatibilidade
PPOWithTrustRegion = PPOAgent
