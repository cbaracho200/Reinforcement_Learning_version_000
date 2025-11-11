"""
Implementação do PPO com Trust Region.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import numpy as np
from typing import Dict, Any, Tuple

from ...core.agent import Agent, PolicyNetwork, ValueNetwork, Decision


class PPOAgent(Agent):
    """Proximal Policy Optimization com Trust Region."""

    def __init__(self, problem_spec, config=None):
        # Configuração padrão do PPO
        default_config = {
            'learning_rate': 3e-4,
            'n_epochs': 10,
            'clip_range': 0.2,
            'clip_range_vf': None,
            'ent_coef': 0.01,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'target_kl': 0.01,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'normalize_advantage': True,
            'use_trust_region': True
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(problem_spec, default_config)

    def _build_networks(self):
        """Constrói redes do PPO."""
        
        # Dimensões
        obs_dim = self.problem_spec._get_obs_dim()
        
        # Verifica tipo de ação
        info = self.problem_spec.get_info()
        self.continuous_actions = info['has_continuous_actions']
        self.discrete_actions = info['has_discrete_actions']
        
        if self.continuous_actions and self.discrete_actions:
            # Ações híbridas
            self._build_hybrid_policy(obs_dim)
        elif self.continuous_actions:
            # Ações contínuas
            self._build_continuous_policy(obs_dim)
        else:
            # Ações discretas
            self._build_discrete_policy(obs_dim)
        
        # Rede de valor
        self.value_net = ValueNetwork(
            obs_dim,
            hidden_dims=[256, 256]
        ).to(self.device)
        
        # Otimizador
        self.optimizer = torch.optim.Adam(
            list(self.policy_net.parameters()) + 
            list(self.value_net.parameters()),
            lr=self.config['learning_rate']
        )

    def _build_discrete_policy(self, obs_dim):
        """Constrói política para ações discretas."""
        
        act_dim = self.problem_spec._get_act_dim()
        
        class DiscretePolicy(PolicyNetwork):
            def __init__(self, obs_dim, act_dim):
                super().__init__(obs_dim, act_dim)
                self.action_head = nn.Linear(self.hidden_dim, act_dim)

            def forward(self, obs):
                features = super().forward(obs)
                logits = self.action_head(features)
                return Categorical(logits=logits)
        
        self.policy_net = DiscretePolicy(obs_dim, act_dim).to(self.device)

    def _build_continuous_policy(self, obs_dim):
        """Constrói política para ações contínuas."""
        
        act_dim = self.problem_spec._get_act_dim()
        
        class ContinuousPolicy(PolicyNetwork):
            def __init__(self, obs_dim, act_dim):
                super().__init__(obs_dim, act_dim)
                self.mean_head = nn.Linear(self.hidden_dim, act_dim)
                self.log_std = nn.Parameter(torch.zeros(act_dim))

            def forward(self, obs):
                features = super().forward(obs)
                mean = self.mean_head(features)
                std = self.log_std.exp()
                return Normal(mean, std)
        
        self.policy_net = ContinuousPolicy(obs_dim, act_dim).to(self.device)

    def _build_hybrid_policy(self, obs_dim):
        """Constrói política para ações híbridas."""
        
        class HybridPolicy(PolicyNetwork):
            def __init__(self, obs_dim, action_spec):
                # Calcula dimensões corretamente
                discrete_dims = {}
                continuous_dims = 0
                
                for name, spec in action_spec.items():
                    if spec['type'] == 'discrete':
                        discrete_dims[name] = spec['n']
                    else:
                        continuous_dims += 1
                
                # CORREÇÃO: usa obs_dim diretamente, não calcula errado
                super().__init__(obs_dim, continuous_dims)
                
                # Heads discretas
                self.discrete_heads = nn.ModuleDict({
                    name: nn.Linear(self.hidden_dim, dim)
                    for name, dim in discrete_dims.items()
                })
                
                # Heads contínuas
                if continuous_dims > 0:
                    self.continuous_mean = nn.Linear(
                        self.hidden_dim, continuous_dims
                    )
                    self.continuous_log_std = nn.Parameter(
                        torch.zeros(continuous_dims)
                    )
                
                self.discrete_dims = discrete_dims
                self.continuous_dims = continuous_dims

            def forward(self, obs):
                features = super().forward(obs)
                
                distributions = {}
                
                # Distribuições discretas
                for name, head in self.discrete_heads.items():
                    logits = head(features)
                    distributions[name] = Categorical(logits=logits)
                
                # Distribuição contínua
                if self.continuous_dims > 0:
                    mean = self.continuous_mean(features)
                    std = self.continuous_log_std.exp()
                    distributions['continuous'] = Normal(mean, std)
                
                return distributions
        
        self.policy_net = HybridPolicy(
            obs_dim, 
            self.problem_spec.action_spec
        ).to(self.device)

    def act(self, observation: np.ndarray, 
            deterministic: bool = False) -> Decision:
        """Seleciona ação."""
        
        # Converte para tensor
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Política
            if hasattr(self.policy_net, 'discrete_heads'):
                # Ações híbridas
                distributions = self.policy_net(obs_tensor)
                
                action = {}
                log_prob = 0
                entropy = 0
                
                for name, dist in distributions.items():
                    if deterministic:
                        if isinstance(dist, Categorical):
                            a = dist.probs.argmax(dim=-1)
                        else:
                            a = dist.mean
                    else:
                        a = dist.sample()
                    
                    action[name] = a.cpu().numpy()[0]
                    log_prob += dist.log_prob(a).sum()
                    entropy += dist.entropy().sum()
                
            else:
                # Ações simples
                dist = self.policy_net(obs_tensor)
                
                if deterministic:
                    if isinstance(dist, Categorical):
                        action = dist.probs.argmax(dim=-1)
                    else:
                        action = dist.mean
                else:
                    action = dist.sample()
                
                log_prob = dist.log_prob(action).sum()
                entropy = dist.entropy().sum()
                action = action.cpu().numpy()[0]
            
            # Valor
            value = self.value_net(obs_tensor)
            
        return Decision(
            action=action,
            value=value.item(),
            log_prob=log_prob.item(),
            entropy=entropy.item(),
            confidence=None
        )

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Atualiza política e valor."""
        
        states = batch['states']
        actions = batch['actions']
        old_log_probs = batch['log_probs']
        advantages = batch['advantages']
        returns = batch['returns']
        
        # Normaliza vantagens
        if self.config['normalize_advantage']:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Múltiplas épocas
        policy_losses = []
        value_losses = []
        entropy_losses = []
        kl_divs = []
        
        for epoch in range(self.config['n_epochs']):
            # Forward pass
            if hasattr(self.policy_net, 'discrete_heads'):
                # Híbrido
                distributions = self.policy_net(states)
                # TODO: Calcular log_prob para ações híbridas
                log_probs = old_log_probs  # Placeholder
                entropy = sum(d.entropy().mean() for d in distributions.values())
            else:
                dist = self.policy_net(states)
                log_probs = dist.log_prob(actions).sum(dim=-1)
                entropy = dist.entropy().mean()
            
            values = self.value_net(states)
            
            # Ratio para PPO
            ratio = (log_probs - old_log_probs).exp()
            
            # Perda da política com clipping
            surr1 = ratio * advantages
            surr2 = torch.clamp(
                ratio, 
                1.0 - self.config['clip_range'],
                1.0 + self.config['clip_range']
            ) * advantages
            
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Perda do valor
            if self.config['clip_range_vf']:
                # Clipping do valor
                values_clipped = batch['values'] + torch.clamp(
                    values - batch['values'],
                    -self.config['clip_range_vf'],
                    self.config['clip_range_vf']
                )
                value_loss_1 = (values - returns) ** 2
                value_loss_2 = (values_clipped - returns) ** 2
                value_loss = torch.max(value_loss_1, value_loss_2).mean()
            else:
                value_loss = F.mse_loss(values, returns)
            
            # Perda de entropia
            entropy_loss = -entropy
            
            # Perda total
            loss = (policy_loss + 
                   self.config['vf_coef'] * value_loss + 
                   self.config['ent_coef'] * entropy_loss)
            
            # KL divergence para early stopping
            with torch.no_grad():
                kl_div = (old_log_probs - log_probs).mean()
                kl_divs.append(kl_div.item())
            
            # Early stopping se KL muito grande
            if self.config['target_kl'] and kl_div > self.config['target_kl']:
                break
            
            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config['max_grad_norm']:
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy_net.parameters()) + 
                    list(self.value_net.parameters()),
                    self.config['max_grad_norm']
                )
            
            self.optimizer.step()
            
            # Registra perdas
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropy_losses.append(entropy_loss.item())
        
        # Métricas
        metrics = {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': -np.mean(entropy_losses),
            'kl_divergence': np.mean(kl_divs),
            'ppo_epochs': epoch + 1
        }
        
        # Atualiza métricas internas
        for key, value in metrics.items():
            if key in self.metrics:
                self.metrics[key].append(value)
        
        return metrics


class PPOWithTrustRegion(PPOAgent):
    """PPO com otimização Trust Region explícita."""

    def __init__(self, problem_spec, config=None):
        super().__init__(problem_spec, config)
        
        # Parâmetros trust region
        self.max_kl = config.get('max_kl', 0.01)
        self.damping = config.get('damping', 0.1)
        self.use_natural_gradient = config.get('use_natural_gradient', True)

    def _conjugate_gradient(self, Avp, b, max_iter=10, tol=1e-10):
        """Resolve Ax = b usando gradiente conjugado."""
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)
        
        for i in range(max_iter):
            Ap = Avp(p)
            alpha = rdotr / (torch.dot(p, Ap) + tol)
            x += alpha * p
            r -= alpha * Ap
            new_rdotr = torch.dot(r, r)
            
            if new_rdotr < tol:
                break
                
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
            
        return x

    def _fisher_vector_product(self, states, p):
        """Calcula produto Fisher-vetor para gradiente natural."""
        
        # Calcula KL divergence
        with torch.enable_grad():
            dist_old = self.policy_net(states)
            
            # Perturbação pequena para calcular KL
            kl = self._kl_divergence(dist_old, dist_old)
            
            # Gradiente do KL
            grads = torch.autograd.grad(
                kl, self.policy_net.parameters(), 
                create_graph=True
            )
            flat_grad = torch.cat([g.view(-1) for g in grads])
            
            # Produto gradiente-vetor
            gvp = (flat_grad * p).sum()
            
            # Segunda derivada (Hessiana)
            fvp_grads = torch.autograd.grad(
                gvp, self.policy_net.parameters()
            )
            fvp = torch.cat([g.view(-1) for g in fvp_grads])
            
        return fvp + self.damping * p

    def _kl_divergence(self, dist1, dist2):
        """Calcula KL divergence entre distribuições."""
        if isinstance(dist1, Categorical):
            return (dist1.probs * (dist1.logits - dist2.logits)).sum(-1).mean()
        elif isinstance(dist1, Normal):
            return torch.distributions.kl_divergence(dist1, dist2).sum(-1).mean()
        else:
            raise NotImplementedError

    def _line_search(self, states, actions, advantages, 
                     step_dir, max_kl, max_iter=10):
        """Line search para encontrar tamanho de passo."""
        
        old_params = torch.cat([
            p.data.view(-1) for p in self.policy_net.parameters()
        ])
        
        old_dist = self.policy_net(states)
        with torch.no_grad():
            old_loss = self._compute_policy_loss(
                states, actions, advantages, old_dist
            )
        
        for stepfrac in [0.5**i for i in range(max_iter)]:
            # Atualiza parâmetros
            new_params = old_params + stepfrac * step_dir
            self._set_flat_params(new_params)
            
            # Verifica KL constraint
            new_dist = self.policy_net(states)
            kl = self._kl_divergence(old_dist, new_dist)
            
            if kl < max_kl:
                # Verifica melhoria
                with torch.no_grad():
                    new_loss = self._compute_policy_loss(
                        states, actions, advantages, new_dist
                    )
                
                if new_loss < old_loss:
                    return stepfrac
        
        # Restaura parâmetros originais se falhar
        self._set_flat_params(old_params)
        return 0

    def _set_flat_params(self, flat_params):
        """Define parâmetros a partir de vetor."""
        idx = 0
        for p in self.policy_net.parameters():
            flat_size = p.numel()
            p.data = flat_params[idx:idx + flat_size].view(p.shape)
            idx += flat_size

    def _compute_policy_loss(self, states, actions, advantages, dist):
        """Calcula perda da política."""
        log_probs = dist.log_prob(actions).sum(dim=-1)
        return -(log_probs * advantages).mean()