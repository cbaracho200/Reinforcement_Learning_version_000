"""
Métodos para otimização com restrições.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass


@dataclass
class ConstraintViolation:
    """Informação sobre violação de restrição."""
    name: str
    value: float
    threshold: float
    violation: float
    satisfied: bool


class LagrangianOptimizer:
    """Otimizador Lagrangiano para restrições."""
    
    def __init__(self,
                 constraint_thresholds: Dict[str, float],
                 learning_rate: float = 0.01,
                 init_penalty: float = 1.0,
                 max_penalty: float = 1000.0,
                 penalty_scale_factor: float = 2.0):
        """
        Args:
            constraint_thresholds: Limites para cada restrição
            learning_rate: Taxa de aprendizado para multiplicadores
            init_penalty: Penalidade inicial
            max_penalty: Penalidade máxima
            penalty_scale_factor: Fator de escala para penalidades
        """
        
        self.thresholds = constraint_thresholds
        self.lr = learning_rate
        self.max_penalty = max_penalty
        self.scale_factor = penalty_scale_factor
        
        # Multiplicadores de Lagrange
        self.lambdas = {
            name: torch.tensor(init_penalty, requires_grad=True)
            for name in constraint_thresholds
        }
        
        # Otimizador para lambdas
        self.lambda_optimizer = torch.optim.Adam(
            list(self.lambdas.values()),
            lr=learning_rate
        )
        
        # Estatísticas
        self.violations_history = {name: [] for name in constraint_thresholds}
        self.lambdas_history = {name: [] for name in constraint_thresholds}
    
    def compute_penalties(self,
                         constraint_values: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calcula penalidades totais para violações.
        
        Args:
            constraint_values: Valores atuais das restrições
            
        Returns:
            penalty: Penalidade total
        """
        
        total_penalty = torch.tensor(0.0)
        violations = {}
        
        for name, value in constraint_values.items():
            if name not in self.thresholds:
                continue
            
            threshold = self.thresholds[name]
            
            # Violação (positiva se viola)
            violation = value - threshold
            violations[name] = violation
            
            # Penalidade Lagrangiana
            if violation > 0:
                penalty = self.lambdas[name] * violation
                total_penalty = total_penalty + penalty
        
        return total_penalty, violations
    
    def update_multipliers(self,
                          violations: Dict[str, torch.Tensor]):
        """Atualiza multiplicadores de Lagrange."""
        
        # Gradiente ascendente nos lambdas
        # (queremos maximizar o Lagrangiano em relação aos lambdas)
        
        self.lambda_optimizer.zero_grad()
        
        for name, violation in violations.items():
            if name in self.lambdas:
                # Lambda deve aumentar se há violação
                lambda_loss = -self.lambdas[name] * violation
                lambda_loss.backward(retain_graph=True)
        
        self.lambda_optimizer.step()
        
        # Clamp lambdas para evitar valores negativos ou muito grandes
        for name, lambda_param in self.lambdas.items():
            lambda_param.data = torch.clamp(
                lambda_param.data, 0, self.max_penalty
            )
            
            # Registra histórico
            self.lambdas_history[name].append(lambda_param.item())
            if name in violations:
                self.violations_history[name].append(violations[name].item())
    
    def get_augmented_reward(self,
                            reward: torch.Tensor,
                            constraint_values: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calcula recompensa aumentada com penalidades.
        
        Args:
            reward: Recompensa original
            constraint_values: Valores das restrições
            
        Returns:
            augmented_reward: Recompensa com penalidades
        """
        
        penalty, violations = self.compute_penalties(constraint_values)
        return reward - penalty
    
    def check_violations(self,
                        constraint_values: Dict[str, float]) -> List[ConstraintViolation]:
        """Verifica quais restrições estão violadas."""
        
        violations = []
        
        for name, value in constraint_values.items():
            if name not in self.thresholds:
                continue
            
            threshold = self.thresholds[name]
            violation_amount = max(0, value - threshold)
            
            violations.append(ConstraintViolation(
                name=name,
                value=value,
                threshold=threshold,
                violation=violation_amount,
                satisfied=violation_amount == 0
            ))
        
        return violations
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Retorna diagnósticos do otimizador."""
        
        diagnostics = {}
        
        for name in self.thresholds:
            if self.violations_history[name]:
                recent_violations = self.violations_history[name][-100:]
                diagnostics[f'constraint/{name}/violation_mean'] = np.mean(recent_violations)
                diagnostics[f'constraint/{name}/violation_max'] = np.max(recent_violations)
                diagnostics[f'constraint/{name}/satisfaction_rate'] = np.mean(
                    [v <= 0 for v in recent_violations]
                )
            
            if self.lambdas_history[name]:
                diagnostics[f'constraint/{name}/lambda'] = self.lambdas_history[name][-1]
        
        return diagnostics


class BarrierMethod:
    """Método de barreira para restrições."""
    
    def __init__(self,
                 barrier_coeff: float = 0.1,
                 decay_rate: float = 0.99):
        """
        Args:
            barrier_coeff: Coeficiente inicial da barreira
            decay_rate: Taxa de decaimento do coeficiente
        """
        
        self.barrier_coeff = barrier_coeff
        self.decay_rate = decay_rate
        self.iteration = 0
    
    def compute_barrier(self,
                       constraint_values: Dict[str, torch.Tensor],
                       constraint_bounds: Dict[str, Tuple[float, float]]) -> torch.Tensor:
        """
        Calcula função barreira logarítmica.
        
        Args:
            constraint_values: Valores atuais
            constraint_bounds: (min, max) para cada restrição
            
        Returns:
            barrier: Valor da função barreira
        """
        
        barrier = torch.tensor(0.0)
        current_coeff = self.barrier_coeff * (self.decay_rate ** self.iteration)
        
        for name, value in constraint_values.items():
            if name not in constraint_bounds:
                continue
            
            min_val, max_val = constraint_bounds[name]
            
            # Barreira logarítmica
            if min_val is not None:
                # -log(x - min)
                margin = value - min_val
                if margin > 0:
                    barrier -= current_coeff * torch.log(margin + 1e-8)
                else:
                    barrier += 1e6  # Penalidade muito alta
            
            if max_val is not None:
                # -log(max - x)
                margin = max_val - value
                if margin > 0:
                    barrier -= current_coeff * torch.log(margin + 1e-8)
                else:
                    barrier += 1e6
        
        return barrier
    
    def step(self):
        """Avança uma iteração (decai coeficiente)."""
        self.iteration += 1


class ProjectedGradient:
    """Gradiente projetado para restrições convexas."""
    
    def __init__(self, projection_fn: Callable):
        """
        Args:
            projection_fn: Função que projeta no conjunto viável
        """
        self.projection_fn = projection_fn
    
    def project(self, params: torch.Tensor) -> torch.Tensor:
        """Projeta parâmetros no conjunto viável."""
        return self.projection_fn(params)
    
    def projected_gradient_step(self,
                               params: torch.Tensor,
                               gradient: torch.Tensor,
                               step_size: float) -> torch.Tensor:
        """
        Executa passo de gradiente projetado.
        
        Args:
            params: Parâmetros atuais
            gradient: Gradiente
            step_size: Tamanho do passo
            
        Returns:
            new_params: Parâmetros atualizados e projetados
        """
        
        # Passo de gradiente
        updated = params - step_size * gradient
        
        # Projeção
        projected = self.project(updated)
        
        return projected