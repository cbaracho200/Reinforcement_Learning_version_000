"""
Implementação de métodos Trust Region para otimização de políticas.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Callable, Tuple, Optional


class TrustRegionOptimizer:
    """Otimizador Trust Region com gradiente natural."""
    
    def __init__(self,
                 max_kl: float = 0.01,
                 damping: float = 0.1,
                 max_iter_cg: int = 10,
                 max_backtracks: int = 10,
                 accept_ratio: float = 0.1,
                 use_natural_gradient: bool = True):
        """
        Args:
            max_kl: KL divergence máximo permitido
            damping: Coeficiente de damping para Fisher
            max_iter_cg: Iterações máximas para gradiente conjugado
            max_backtracks: Tentativas máximas de line search
            accept_ratio: Taxa mínima de aceitação para line search
            use_natural_gradient: Se usa gradiente natural
        """
        
        self.max_kl = max_kl
        self.damping = damping
        self.max_iter_cg = max_iter_cg
        self.max_backtracks = max_backtracks
        self.accept_ratio = accept_ratio
        self.use_natural_gradient = use_natural_gradient
        
        # Estatísticas
        self.stats = {
            'n_updates': 0,
            'n_backtracks': [],
            'kl_divergences': [],
            'improvements': []
        }
    
    def step(self,
             policy: nn.Module,
             get_loss: Callable,
             get_kl: Callable,
             max_step_size: Optional[float] = None) -> Tuple[float, dict]:
        """
        Executa um passo de otimização trust region.
        
        Args:
            policy: Rede da política
            get_loss: Função que retorna loss(policy)
            get_kl: Função que retorna KL(policy_old, policy)
            max_step_size: Tamanho máximo do passo
            
        Returns:
            improvement: Melhoria no objetivo
            info: Informações sobre o passo
        """
        
        # Calcula loss e gradiente inicial
        loss_before = get_loss()
        loss_before.backward(retain_graph=True)
        
        # Extrai gradiente
        grads = []
        for param in policy.parameters():
            grads.append(param.grad.view(-1))
        
        flat_grad = torch.cat(grads)
        
        # Direção de busca
        if self.use_natural_gradient:
            # Gradiente natural via Fisher
            search_dir = self._natural_gradient_step(
                policy, flat_grad, get_kl
            )
        else:
            # Gradiente vanilla
            search_dir = -flat_grad
        
        # Calcula tamanho máximo do passo
        if max_step_size is None:
            max_step_size = self._compute_max_step_size(
                search_dir, get_kl, policy
            )
        
        # Line search
        step_size, n_backtracks = self._line_search(
            policy, search_dir, get_loss, get_kl,
            max_step_size, loss_before
        )
        
        # Calcula melhoria
        with torch.no_grad():
            loss_after = get_loss()
            improvement = (loss_before - loss_after).item()
            final_kl = get_kl().item()
        
        # Atualiza estatísticas
        self.stats['n_updates'] += 1
        self.stats['n_backtracks'].append(n_backtracks)
        self.stats['kl_divergences'].append(final_kl)
        self.stats['improvements'].append(improvement)
        
        info = {
            'step_size': step_size,
            'n_backtracks': n_backtracks,
            'improvement': improvement,
            'kl_divergence': final_kl,
            'loss_before': loss_before.item(),
            'loss_after': loss_after.item()
        }
        
        return improvement, info
    
    def _natural_gradient_step(self,
                               policy: nn.Module,
                               flat_grad: torch.Tensor,
                               get_kl: Callable) -> torch.Tensor:
        """Calcula direção do gradiente natural."""
        
        def fisher_vector_product(v):
            """Produto matriz Fisher × vetor."""
            # Calcula Hv onde H é a Hessiana do KL
            kl = get_kl()
            
            # Primeira derivada
            grads = torch.autograd.grad(
                kl, policy.parameters(), 
                create_graph=True
            )
            flat_grad_kl = torch.cat([g.view(-1) for g in grads])
            
            # Segunda derivada (produto com v)
            grad_prod = (flat_grad_kl * v).sum()
            grads2 = torch.autograd.grad(
                grad_prod, policy.parameters()
            )
            flat_grad2 = torch.cat([g.view(-1) for g in grads2])
            
            return flat_grad2 + self.damping * v
        
        # Resolve F^{-1} g usando gradiente conjugado
        search_dir = self._conjugate_gradient(
            fisher_vector_product, 
            -flat_grad,
            max_iter=self.max_iter_cg
        )
        
        return search_dir
    
    def _conjugate_gradient(self,
                           Avp: Callable,
                           b: torch.Tensor,
                           max_iter: int = 10,
                           tol: float = 1e-10) -> torch.Tensor:
        """
        Resolve Ax = b usando gradiente conjugado.
        
        Args:
            Avp: Função que calcula produto A × vetor
            b: Vetor do lado direito
            max_iter: Iterações máximas
            tol: Tolerância
            
        Returns:
            x: Solução aproximada
        """
        
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        rdotr = r.dot(r)
        
        for i in range(max_iter):
            Ap = Avp(p)
            alpha = rdotr / (p.dot(Ap) + tol)
            
            x += alpha * p
            r -= alpha * Ap
            
            new_rdotr = r.dot(r)
            if new_rdotr < tol:
                break
            
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
        
        return x
    
    def _compute_max_step_size(self,
                              search_dir: torch.Tensor,
                              get_kl: Callable,
                              policy: nn.Module) -> float:
        """Calcula tamanho máximo do passo para satisfazer constraint KL."""
        
        # Aproximação quadrática do KL
        # KL(θ + α*d) ≈ 0.5 * α^2 * d^T H d
        
        def fisher_vector_product(v):
            kl = get_kl()
            grads = torch.autograd.grad(
                kl, policy.parameters(), 
                create_graph=True
            )
            flat_grad_kl = torch.cat([g.view(-1) for g in grads])
            grad_prod = (flat_grad_kl * v).sum()
            grads2 = torch.autograd.grad(
                grad_prod, policy.parameters()
            )
            return torch.cat([g.view(-1) for g in grads2])
        
        Hd = fisher_vector_product(search_dir)
        dHd = search_dir.dot(Hd)
        
        # α_max = sqrt(2 * KL_max / d^T H d)
        max_step = torch.sqrt(2 * self.max_kl / (dHd + 1e-8))
        
        return max_step.item()
    
    def _line_search(self,
                    policy: nn.Module,
                    search_dir: torch.Tensor,
                    get_loss: Callable,
                    get_kl: Callable,
                    max_step_size: float,
                    expected_improve: torch.Tensor) -> Tuple[float, int]:
        """
        Line search com backtracking.
        
        Returns:
            step_size: Tamanho do passo aceito
            n_backtracks: Número de backtracks realizados
        """
        
        # Salva parâmetros originais
        old_params = []
        for param in policy.parameters():
            old_params.append(param.data.clone())
        
        # Tenta diferentes tamanhos de passo
        step_sizes = max_step_size * (0.5 ** np.arange(self.max_backtracks))
        
        for n_backtracks, step_size in enumerate(step_sizes):
            # Atualiza parâmetros
            self._update_params(policy, search_dir, step_size)
            
            # Verifica constraint KL
            with torch.no_grad():
                kl = get_kl()
                
            if kl > self.max_kl * 1.5:
                # KL muito grande, continua backtracking
                self._restore_params(policy, old_params)
                continue
            
            # Verifica melhoria
            with torch.no_grad():
                loss = get_loss()
                actual_improve = expected_improve - loss
                improve_ratio = actual_improve / expected_improve
            
            if improve_ratio > self.accept_ratio and kl < self.max_kl:
                # Aceita o passo
                return step_size, n_backtracks
            
            # Restaura e tenta menor passo
            self._restore_params(policy, old_params)
        
        # Falhou em encontrar bom passo, restaura original
        self._restore_params(policy, old_params)
        return 0.0, self.max_backtracks
    
    def _update_params(self,
                      policy: nn.Module,
                      direction: torch.Tensor,
                      step_size: float):
        """Atualiza parâmetros da política."""
        idx = 0
        for param in policy.parameters():
            size = param.numel()
            param.data += step_size * direction[idx:idx+size].view(param.shape)
            idx += size
    
    def _restore_params(self,
                       policy: nn.Module,
                       old_params: list):
        """Restaura parâmetros originais."""
        for param, old_param in zip(policy.parameters(), old_params):
            param.data = old_param
    
    def get_diagnostics(self) -> dict:
        """Retorna diagnósticos do otimizador."""
        if not self.stats['n_backtracks']:
            return {}
        
        return {
            'trust_region/mean_backtracks': np.mean(self.stats['n_backtracks']),
            'trust_region/mean_kl': np.mean(self.stats['kl_divergences']),
            'trust_region/mean_improvement': np.mean(self.stats['improvements']),
            'trust_region/n_updates': self.stats['n_updates']
        }