"""
Implementação do Soft Actor-Critic (SAC).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from typing import Dict, Any, Tuple
import copy

from ...core.agent import Agent, Decision


class SACAgent(Agent):
    """Soft Actor-Critic para ações contínuas."""
    
    def __init__(self, problem_spec, config=None):
        # Configuração padrão do SAC
        default_config = {
            'learning_rate': 3e-4,
            'batch_size': 256,
            'tau': 0.005,  # Soft update
            'gamma': 0.99,
            'alpha': 0.2,  # Temperatura da entropia
            'auto_entropy': True,
            'target_entropy': None,
            'hidden_dims': [256, 256],
            'activation': 'relu'
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(problem_spec, default_config)
    
    def _build_networks(self):
        """Constrói redes do SAC."""
        
        obs_dim = self.problem_spec._get_obs_dim()
        act_dim = self.problem_spec._get_act_dim()
        
        # Verifica se tem apenas ações contínuas
        info = self.problem_spec.get_info()
        if not info['has_continuous_actions'] or info['has_discrete_actions']:
            raise ValueError("SAC suporta apenas ações contínuas")
        
        # Actor (política)
        self.actor = Actor(
            obs_dim, act_dim,
            hidden_dims=self.config['hidden_dims'],
            activation=self.config['activation']
        ).to(self.device)
        
        # Critics (duas Q-functions para reduzir overestimation)
        self.critic1 = Critic(
            obs_dim, act_dim,
            hidden_dims=self.config['hidden_dims'],
            activation=self.config['activation']
        ).to(self.device)
        
        self.critic2 = Critic(
            obs_dim, act_dim,
            hidden_dims=self.config['hidden_dims'],
            activation=self.config['activation']
        ).to(self.device)
        
        # Target networks
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)
        
        # Entropia automática
        if self.config['auto_entropy']:
            # Target entropy = -dim(A)
            if self.config['target_entropy'] is None:
                self.target_entropy = -act_dim
            else:
                self.target_entropy = self.config['target_entropy']
            
            # Log alpha aprendível
            self.log_alpha = torch.tensor(
                np.log(self.config['alpha']),
                requires_grad=True,
                device=self.device
            )
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = torch.optim.Adam(
                [self.log_alpha],
                lr=self.config['learning_rate']
            )
        else:
            self.alpha = self.config['alpha']
        
        # Otimizadores
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.config['learning_rate']
        )
        
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=self.config['learning_rate']
        )
        
        # Registra target networks
        self.target_nets = {
            'critic1': self.critic1_target,
            'critic2': self.critic2_target
        }
    
    def act(self, observation: np.ndarray, 
            deterministic: bool = False) -> Decision:
        """Seleciona ação usando a política."""
        
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if deterministic:
                _, _, action = self.actor(obs_tensor, deterministic=True)
            else:
                action, log_prob, _ = self.actor(obs_tensor)
            
            # Calcula valor Q
            q1 = self.critic1(obs_tensor, action)
            q2 = self.critic2(obs_tensor, action)
            q_value = torch.min(q1, q2)
        
        return Decision(
            action=action.cpu().numpy()[0],
            value=q_value.item(),
            log_prob=log_prob.item() if not deterministic else None,
            confidence=None
        )
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Atualiza actor e critics."""
        
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards'].unsqueeze(-1)
        next_states = batch['next_states']
        dones = batch['dones'].unsqueeze(-1)
        
        # Update critics
        with torch.no_grad():
            # Amostra ação seguinte
            next_actions, next_log_probs, _ = self.actor(next_states)
            
            # Target Q values
            target_q1 = self.critic1_target(next_states, next_actions)
            target_q2 = self.critic2_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            
            # Adiciona bônus de entropia
            target_q = target_q - self.alpha * next_log_probs
            
            # Bellman backup
            target_value = rewards + (1 - dones) * self.config['gamma'] * target_q
        
        # Q-function losses
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        
        critic1_loss = F.mse_loss(q1, target_value)
        critic2_loss = F.mse_loss(q2, target_value)
        critic_loss = critic1_loss + critic2_loss
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        new_actions, log_probs, _ = self.actor(states)
        
        q1_new = self.critic1(states, new_actions)
        q2_new = self.critic2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        # Maximiza Q - alpha * entropy
        actor_loss = (self.alpha * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update alpha (temperatura)
        alpha_loss = 0
        if self.config['auto_entropy']:
            alpha_loss = -(self.log_alpha * 
                          (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp()
        
        # Soft update das target networks
        self._soft_update()
        
        # Métricas
        metrics = {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item() if self.config['auto_entropy'] else 0,
            'alpha': self.alpha.item() if self.config['auto_entropy'] else self.alpha,
            'mean_q': q_new.mean().item(),
            'mean_entropy': -log_probs.mean().item()
        }
        
        # TD errors para prioritized replay
        with torch.no_grad():
            td_error1 = torch.abs(q1 - target_value).squeeze()
            td_error2 = torch.abs(q2 - target_value).squeeze()
            metrics['td_errors'] = torch.max(td_error1, td_error2)
        
        return metrics
    
    def _soft_update(self):
        """Soft update das target networks."""
        tau = self.config['tau']
        
        for param, target_param in zip(
            self.critic1.parameters(),
            self.critic1_target.parameters()
        ):
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data
            )
        
        for param, target_param in zip(
            self.critic2.parameters(),
            self.critic2_target.parameters()
        ):
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data
            )


class Actor(nn.Module):
    """Rede do ator (política) para SAC."""
    
    def __init__(self, obs_dim, act_dim, hidden_dims=[256, 256], 
                 activation='relu'):
        super().__init__()
        
        # Função de ativação
        act_fn = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'elu': nn.ELU
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
        
        # Heads para média e log_std
        self.mean_head = nn.Linear(prev_dim, act_dim)
        self.log_std_head = nn.Linear(prev_dim, act_dim)
        
        # Limites para log_std
        self.log_std_min = -20
        self.log_std_max = 2
    
    def forward(self, obs, deterministic=False):
        """Forward pass do ator."""
        features = self.trunk(obs)
        
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()
        
        # Distribuição Normal
        dist = Normal(mean, std)
        
        if deterministic:
            action = mean
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        else:
            # Reparametrization trick
            action = dist.rsample()
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
            
            # Correção para tanh squashing
            log_prob -= (2 * (np.log(2) - action - 
                            F.softplus(-2 * action))).sum(dim=-1, keepdim=True)
        
        # Squash para [-1, 1]
        action = torch.tanh(action)
        
        return action, log_prob, mean


class Critic(nn.Module):
    """Rede do crítico (Q-function) para SAC."""
    
    def __init__(self, obs_dim, act_dim, hidden_dims=[256, 256], 
                 activation='relu'):
        super().__init__()
        
        # Função de ativação
        act_fn = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'elu': nn.ELU
        }[activation]
        
        # Constrói MLP
        layers = []
        prev_dim = obs_dim + act_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                act_fn(),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, obs, action):
        """Calcula Q(s, a)."""
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)