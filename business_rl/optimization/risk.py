"""
Métodos para gerenciamento de risco em RL.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class RiskMetrics:
    """Métricas de risco."""
    var: float  # Value at Risk
    cvar: float  # Conditional Value at Risk
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    tail_ratio: float


class CVaROptimizer:
    """Otimizador para Conditional Value at Risk."""
    
    def __init__(self,
                 alpha: float = 0.1,
                 sample_size: int = 100):
        """
        Args:
            alpha: Nível de confiança (ex: 0.1 = CVaR dos 10% piores)
            sample_size: Tamanho da amostra para estimação
        """
        
        self.alpha = alpha
        self.sample_size = sample_size
        
        # Variável auxiliar para CVaR (VaR)
        self.var = torch.tensor(0.0, requires_grad=True)
        self.var_optimizer = torch.optim.Adam([self.var], lr=0.01)
        
        # Histórico
        self.returns_buffer = []
        self.cvar_history = []
        self.var_history = []
    
    def compute_cvar_loss(self,
                         returns: torch.Tensor,
                         weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calcula loss do CVaR para otimização.
        
        Args:
            returns: Retornos (negativos para perdas)
            weights: Pesos opcionais para amostras
            
        Returns:
            cvar_loss: Loss para minimizar CVaR
        """
        
        batch_size = returns.shape[0]
        
        if weights is None:
            weights = torch.ones_like(returns) / batch_size
        
        # CVaR = VaR + (1/α) * E[(VaR - R)+]
        # Onde (x)+ = max(x, 0)
        
        excess_loss = torch.relu(self.var - returns)
        cvar = self.var + (1 / self.alpha) * (weights * excess_loss).sum()
        
        return cvar
    
    def update_var(self, returns: torch.Tensor):
        """Atualiza estimativa do VaR."""
        
        # Gradiente ascendente no VaR
        # (queremos encontrar o quantil correto)
        
        self.var_optimizer.zero_grad()
        
        # Loss para encontrar o quantil α
        indicator = (returns <= self.var).float()
        var_loss = torch.abs(indicator.mean() - self.alpha)
        
        var_loss.backward()
        self.var_optimizer.step()
        
        # Registra histórico
        self.var_history.append(self.var.item())
        
        # Calcula CVaR empírico
        with torch.no_grad():
            sorted_returns = torch.sort(returns)[0]
            var_idx = int(self.alpha * len(returns))
            empirical_var = sorted_returns[var_idx]
            empirical_cvar = sorted_returns[:var_idx].mean()
            self.cvar_history.append(empirical_cvar.item())
    
    def risk_adjusted_reward(self,
                            reward: torch.Tensor,
                            risk_weight: float = 0.5) -> torch.Tensor:
        """
        Calcula recompensa ajustada ao risco.
        
        Args:
            reward: Recompensa esperada
            risk_weight: Peso do risco (0 = neutro, 1 = máxima aversão)
            
        Returns:
            adjusted_reward: Recompensa ajustada
        """
        
        if len(self.returns_buffer) < self.sample_size:
            # Não há dados suficientes, retorna recompensa normal
            return reward
        
        # Amostra retornos históricos
        sample_idx = np.random.choice(
            len(self.returns_buffer),
            min(self.sample_size, len(self.returns_buffer))
        )
        sample_returns = torch.tensor(
            [self.returns_buffer[i] for i in sample_idx]
        )
        
        # Calcula CVaR da amostra
        cvar = self.compute_cvar_loss(sample_returns)
        
        # Ajusta recompensa
        # R_adj = (1 - w) * R_expected + w * (-CVaR)
        adjusted = (1 - risk_weight) * reward - risk_weight * cvar
        
        return adjusted
    
    def add_returns(self, returns: List[float]):
        """Adiciona retornos ao buffer."""
        self.returns_buffer.extend(returns)
        
        # Limita tamanho do buffer
        if len(self.returns_buffer) > 10000:
            self.returns_buffer = self.returns_buffer[-10000:]
    
    def compute_risk_metrics(self,
                            returns: np.ndarray,
                            risk_free_rate: float = 0.0) -> RiskMetrics:
        """
        Calcula métricas de risco completas.
        
        Args:
            returns: Array de retornos
            risk_free_rate: Taxa livre de risco
            
        Returns:
            metrics: Métricas de risco
        """
        
        if len(returns) < 2:
            return RiskMetrics(0, 0, 0, 0, 0, 0)
        
        # VaR e CVaR
        sorted_returns = np.sort(returns)
        var_idx = int(self.alpha * len(returns))
        var = sorted_returns[var_idx]
        cvar = sorted_returns[:var_idx].mean()
        
        # Drawdown máximo
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Sharpe ratio
        excess_returns = returns - risk_free_rate
        sharpe = excess_returns.mean() / (excess_returns.std() + 1e-8)
        
        # Sortino ratio (apenas volatilidade negativa)
        downside_returns = returns[returns < risk_free_rate]
        if len(downside_returns) > 0:
            downside_std = downside_returns.std()
            sortino = excess_returns.mean() / (downside_std + 1e-8)
        else:
            sortino = sharpe
        
        # Tail ratio
        right_tail = returns[returns > np.percentile(returns, 95)]
        left_tail = returns[returns < np.percentile(returns, 5)]
        
        if len(left_tail) > 0 and left_tail.mean() != 0:
            tail_ratio = right_tail.mean() / abs(left_tail.mean())
        else:
            tail_ratio = float('inf')
        
        return RiskMetrics(
            var=var,
            cvar=cvar,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            tail_ratio=tail_ratio
        )


class DistributionalRL:
    """Métodos distribucionais para RL com foco em risco."""
    
    def __init__(self,
                 n_atoms: int = 51,
                 v_min: float = -10.0,
                 v_max: float = 10.0):
        """
        Args:
            n_atoms: Número de átomos na distribuição
            v_min: Valor mínimo do suporte
            v_max: Valor máximo do suporte
        """
        
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        
        # Suporte da distribuição
        self.delta = (v_max - v_min) / (n_atoms - 1)
        self.support = torch.linspace(v_min, v_max, n_atoms)
    
    def compute_distributional_loss(self,
                                   logits: torch.Tensor,
                                   rewards: torch.Tensor,
                                   next_logits: torch.Tensor,
                                   dones: torch.Tensor,
                                   gamma: float = 0.99) -> torch.Tensor:
        """
        Calcula loss distribucional (C51).
        
        Args:
            logits: Logits da distribuição atual
            rewards: Recompensas
            next_logits: Logits da distribuição seguinte
            dones: Flags de término
            gamma: Fator de desconto
            
        Returns:
            loss: Loss distribucional
        """
        
        batch_size = logits.shape[0]
        
        # Distribuições atuais e seguintes
        probs = torch.softmax(logits, dim=-1)
        next_probs = torch.softmax(next_logits, dim=-1)
        
        # Projeta distribuição de Bellman
        # T𝒵 = r + γ𝒵'
        projected_support = rewards.unsqueeze(-1) + \
                          gamma * self.support.unsqueeze(0) * \
                          (1 - dones).unsqueeze(-1)
        
        # Clamp no suporte
        projected_support = torch.clamp(
            projected_support, self.v_min, self.v_max
        )
        
        # Projeta na grade discreta
        b = (projected_support - self.v_min) / self.delta
        lower = b.floor().long()
        upper = b.ceil().long()
        
        # Distribui probabilidade
        target_probs = torch.zeros_like(probs)
        
        offset = torch.arange(batch_size).unsqueeze(-1) * self.n_atoms
        
        for i in range(self.n_atoms):
            # Probabilidade inferior
            lower_idx = (lower[:, i] + offset).flatten()
            lower_prob = next_probs[:, i] * (upper[:, i] - b[:, i])
            target_probs.view(-1).scatter_add_(
                0, lower_idx, lower_prob.flatten()
            )
            
            # Probabilidade superior
            upper_idx = (upper[:, i] + offset).flatten()
            upper_prob = next_probs[:, i] * (b[:, i] - lower[:, i])
            target_probs.view(-1).scatter_add_(
                0, upper_idx, upper_prob.flatten()
            )
        
        # Cross-entropy loss
        loss = -(target_probs * logits).sum(dim=-1).mean()
        
        return loss
    
    def get_risk_sensitive_q(self,
                            logits: torch.Tensor,
                            risk_measure: str = 'cvar',
                            alpha: float = 0.1) -> torch.Tensor:
        """
        Extrai Q-value sensível ao risco da distribuição.
        
        Args:
            logits: Logits da distribuição
            risk_measure: 'mean', 'var', 'cvar', 'wang'
            alpha: Parâmetro para CVaR
            
        Returns:
            q_values: Q-values ajustados ao risco
        """
        
        probs = torch.softmax(logits, dim=-1)
        
        if risk_measure == 'mean':
            # Q-value esperado
            return (probs * self.support).sum(dim=-1)
        
        elif risk_measure == 'cvar':
            # CVaR dos α% piores outcomes
            cumsum = probs.cumsum(dim=-1)
            idx = (cumsum <= alpha).sum(dim=-1)
            
            # Média ponderada dos piores outcomes
            q_values = []
            for i in range(logits.shape[0]):
                worst_probs = probs[i, :idx[i]+1]
                worst_support = self.support[:idx[i]+1]
                q = (worst_probs * worst_support).sum() / worst_probs.sum()
                q_values.append(q)
            
            return torch.tensor(q_values)
        
        elif risk_measure == 'wang':
            # Wang transform (distorção de probabilidade)
            # Implementação simplificada
            cumsum = probs.cumsum(dim=-1)
            distorted = torch.pow(cumsum, 1 / (1 + alpha))
            distorted_probs = torch.diff(
                distorted, 
                prepend=torch.zeros_like(distorted[:, :1])
            )
            
            return (distorted_probs * self.support).sum(dim=-1)
        
        else:
            raise ValueError(f"Medida de risco desconhecida: {risk_measure}")