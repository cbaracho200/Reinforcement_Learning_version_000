"""
Sistema unificado de memória para armazenamento de experiências.
"""

from collections import deque, namedtuple
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch
from dataclasses import dataclass, field
import random


# Transição básica
Transition = namedtuple('Transition', 
    ['state', 'action', 'reward', 'next_state', 'done', 'info'])


@dataclass
class Experience:
    """Experiência expandida com informações adicionais."""
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool
    truncated: bool = False
    
    # Informações extras
    log_prob: Optional[float] = None
    value: Optional[float] = None
    advantage: Optional[float] = None
    returns: Optional[float] = None
    
    # Múltiplos objetivos
    reward_dict: Dict[str, float] = field(default_factory=dict)
    
    # Informações de risco
    risk_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Metadados
    episode_id: Optional[int] = None
    step_id: Optional[int] = None
    timestamp: Optional[float] = None


class UnifiedMemory:
    """Sistema de memória unificado para todos os algoritmos."""
    
    def __init__(self, 
                 capacity: int = 10000,
                 batch_size: int = 32,
                 prioritized: bool = False,
                 n_step: int = 1,
                 gamma: float = 0.99):
        
        self.capacity = capacity
        self.batch_size = batch_size
        self.prioritized = prioritized
        self.n_step = n_step
        self.gamma = gamma
        
        # Armazenamento principal
        self.buffer = deque(maxlen=capacity)
        
        # Buffer temporário para n-step
        self.n_step_buffer = deque(maxlen=n_step)
        
        # Prioridades para PER
        if prioritized:
            self.priorities = deque(maxlen=capacity)
            self.alpha = 0.6  # Priorização
            self.beta = 0.4   # Correção de bias
            self.beta_increment = 0.001
            self.epsilon = 1e-6
        
        # Estatísticas
        self.total_added = 0
        self.total_sampled = 0
        
        # Cache para episódios completos
        self.current_episode = []
        self.episodes = []
        
    def add(self, 
            state: np.ndarray,
            action: np.ndarray,
            reward: float,
            next_state: np.ndarray,
            done: bool,
            **kwargs):
        """Adiciona uma experiência à memória."""
        
        exp = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            **kwargs
        )
        
        # Adiciona ao episódio atual
        self.current_episode.append(exp)
        
        # Se n-step, processa buffer
        if self.n_step > 1:
            self.n_step_buffer.append(exp)
            
            if len(self.n_step_buffer) == self.n_step:
                n_step_exp = self._compute_n_step_return()
                self._add_to_buffer(n_step_exp)
        else:
            self._add_to_buffer(exp)
        
        # Se episódio terminou
        if done:
            self.episodes.append(self.current_episode)
            self.current_episode = []
            
            # Limpa n-step buffer
            while self.n_step_buffer:
                n_step_exp = self._compute_n_step_return()
                self._add_to_buffer(n_step_exp)
        
        self.total_added += 1
    
    def _add_to_buffer(self, exp: Experience):
        """Adiciona experiência ao buffer principal."""
        self.buffer.append(exp)
        
        if self.prioritized:
            # Nova experiência recebe prioridade máxima
            max_priority = max(self.priorities, default=1.0)
            self.priorities.append(max_priority)
    
    def _compute_n_step_return(self) -> Experience:
        """Calcula retorno n-step."""
        if not self.n_step_buffer:
            return None
        
        # Primeira transição
        first = self.n_step_buffer[0]
        
        # Calcula retorno descontado
        n_step_return = 0.0
        for i, exp in enumerate(self.n_step_buffer):
            n_step_return += (self.gamma ** i) * exp.reward
            if exp.done:
                break
        
        # Última transição válida
        last = self.n_step_buffer[-1]
        
        return Experience(
            state=first.state,
            action=first.action,
            reward=n_step_return,
            next_state=last.next_state,
            done=last.done,
            **{k: v for k, v in first.__dict__.items() 
               if k not in ['state', 'action', 'reward', 'next_state', 'done']}
        )
    
    def sample(self, 
               batch_size: Optional[int] = None,
               method: str = 'uniform') -> Dict[str, torch.Tensor]:
        """Amostra batch de experiências."""
        
        batch_size = batch_size or self.batch_size
        
        if method == 'uniform':
            indices = np.random.choice(len(self.buffer), batch_size)
            batch = [self.buffer[i] for i in indices]
            weights = None
            
        elif method == 'prioritized' and self.prioritized:
            batch, weights, indices = self._prioritized_sample(batch_size)
            
        elif method == 'sequential':
            # Para algoritmos on-policy como PPO
            batch = list(self.buffer)[-batch_size:]
            weights = None
            indices = list(range(len(self.buffer) - batch_size, len(self.buffer)))
            
        else:
            raise ValueError(f"Método de amostragem inválido: {method}")
        
        self.total_sampled += len(batch)
        
        # Converte para tensores
        batch_dict = self._batch_to_tensors(batch)
        
        if weights is not None:
            batch_dict['weights'] = torch.FloatTensor(weights)
            batch_dict['indices'] = indices
        
        return batch_dict
    
    def _prioritized_sample(self, batch_size: int) -> Tuple:
        """Amostragem com priorização."""
        if len(self.buffer) == 0:
            return [], None, []
        
        # Calcula probabilidades
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Amostra índices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # Calcula pesos para correção de bias
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Incrementa beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Coleta experiências
        batch = [self.buffer[i] for i in indices]
        
        return batch, weights, indices
    
    def update_priorities(self, indices: List[int], priorities: List[float]):
        """Atualiza prioridades das experiências."""
        if not self.prioritized:
            return
        
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + self.epsilon
    
    def _batch_to_tensors(self, batch: List[Experience]) -> Dict[str, torch.Tensor]:
        """Converte batch para tensores."""

        # Função auxiliar para converter dict em array
        def _dict_to_array(obs):
            if isinstance(obs, dict):
                # Ordena por chave para consistência
                return np.array([obs[k] for k in sorted(obs.keys())], dtype=np.float32)
            return np.array(obs, dtype=np.float32)

        # Agrupa por campo
        states = np.array([_dict_to_array(exp.state) for exp in batch])
        actions = np.array([exp.action for exp in batch])
        rewards = np.array([exp.reward for exp in batch])
        next_states = np.array([_dict_to_array(exp.next_state) for exp in batch])
        dones = np.array([exp.done for exp in batch])
        
        batch_dict = {
            'states': torch.FloatTensor(states),
            'actions': torch.FloatTensor(actions),
            'rewards': torch.FloatTensor(rewards),
            'next_states': torch.FloatTensor(next_states),
            'dones': torch.FloatTensor(dones)
        }
        
        # Adiciona campos opcionais se existirem
        if batch[0].log_prob is not None:
            batch_dict['log_probs'] = torch.FloatTensor(
                [exp.log_prob for exp in batch]
            )
        
        if batch[0].value is not None:
            batch_dict['values'] = torch.FloatTensor(
                [exp.value for exp in batch]
            )
        
        if batch[0].advantage is not None:
            batch_dict['advantages'] = torch.FloatTensor(
                [exp.advantage for exp in batch]
            )
        
        if batch[0].returns is not None:
            batch_dict['returns'] = torch.FloatTensor(
                [exp.returns for exp in batch]
            )
        
        return batch_dict
    
    def compute_returns(self, gamma: float = 0.99,
                       lambda_: float = 0.95) -> None:
        """Calcula retornos e vantagens usando GAE (Generalized Advantage Estimation)."""

        # Processa episódios completos
        for episode in self.episodes:
            if not episode:
                continue

            # GAE: Calcula vantagens e retornos
            advantages = []
            last_advantage = 0

            # Reverso: do fim para o início
            for i in reversed(range(len(episode))):
                exp = episode[i]

                # Próximo valor (bootstrap)
                if i == len(episode) - 1:
                    next_value = 0  # Terminal
                else:
                    next_value = episode[i + 1].value if episode[i + 1].value is not None else 0

                # TD Error (delta)
                value = exp.value if exp.value is not None else 0
                delta = exp.reward + gamma * next_value * (1 - exp.done) - value

                # GAE
                advantage = delta + gamma * lambda_ * (1 - exp.done) * last_advantage
                advantages.insert(0, advantage)
                last_advantage = advantage

            # Retornos = vantagens + valores
            for exp, adv in zip(episode, advantages):
                exp.advantage = adv
                value = exp.value if exp.value is not None else 0
                exp.returns = adv + value

        # IMPORTANTE: Também processa o episódio atual (ainda não terminado)
        # Isso é crucial para PPO que precisa de experiências frequentes
        if self.current_episode:
            advantages = []
            last_advantage = 0

            for i in reversed(range(len(self.current_episode))):
                exp = self.current_episode[i]

                # Próximo valor
                if i == len(self.current_episode) - 1:
                    # Episódio não terminado: usa valor atual como bootstrap
                    next_value = exp.value if exp.value is not None else 0
                else:
                    next_value = self.current_episode[i + 1].value if self.current_episode[i + 1].value is not None else 0

                # TD Error
                value = exp.value if exp.value is not None else 0
                delta = exp.reward + gamma * next_value * (1 - exp.done) - value

                # GAE
                advantage = delta + gamma * lambda_ * (1 - exp.done) * last_advantage
                advantages.insert(0, advantage)
                last_advantage = advantage

            # Atualiza experiências
            for exp, adv in zip(self.current_episode, advantages):
                exp.advantage = adv
                value = exp.value if exp.value is not None else 0
                exp.returns = adv + value
    
    def clear(self):
        """Limpa a memória."""
        self.buffer.clear()
        self.current_episode.clear()
        self.episodes.clear()
        self.n_step_buffer.clear()
        
        if self.prioritized:
            self.priorities.clear()
    
    def __len__(self) -> int:
        """Retorna tamanho atual do buffer."""
        return len(self.buffer)
    
    def is_ready(self) -> bool:
        """Verifica se há experiências suficientes para amostrar."""
        return len(self.buffer) >= self.batch_size
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas da memória."""
        return {
            'size': len(self.buffer),
            'capacity': self.capacity,
            'utilization': len(self.buffer) / self.capacity,
            'total_added': self.total_added,
            'total_sampled': self.total_sampled,
            'episodes_stored': len(self.episodes),
            'current_episode_length': len(self.current_episode)
        }


class RolloutBuffer:
    """Buffer especializado para algoritmos on-policy (PPO)."""
    
    def __init__(self, capacity: int = 2048):
        self.capacity = capacity
        self.reset()
    
    def reset(self):
        """Limpa o buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.advantages = None
        self.returns = None
        self.size = 0
    
    def add(self, state, action, reward, value, log_prob, done):
        """Adiciona transição ao buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.size += 1
    
    def compute_advantages(self, last_value, gamma=0.99, lambda_=0.95):
        """Calcula vantagens usando GAE."""
        advantages = np.zeros(self.size)
        last_advantage = 0
        
        for t in reversed(range(self.size)):
            if t == self.size - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]
            
            delta = (self.rewards[t] + 
                    gamma * next_value * (1 - self.dones[t]) - 
                    self.values[t])
            
            advantages[t] = (delta + 
                           gamma * lambda_ * (1 - self.dones[t]) * 
                           last_advantage)
            last_advantage = advantages[t]
        
        self.advantages = advantages
        self.returns = advantages + np.array(self.values)
    
    def get_batches(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Retorna batches para treinamento."""
        indices = np.random.permutation(self.size)
        
        for start_idx in range(0, self.size, batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            
            yield {
                'states': torch.FloatTensor(
                    np.array([self.states[i] for i in batch_indices])
                ),
                'actions': torch.FloatTensor(
                    np.array([self.actions[i] for i in batch_indices])
                ),
                'log_probs': torch.FloatTensor(
                    np.array([self.log_probs[i] for i in batch_indices])
                ),
                'advantages': torch.FloatTensor(
                    self.advantages[batch_indices]
                ),
                'returns': torch.FloatTensor(
                    self.returns[batch_indices]
                )
            }