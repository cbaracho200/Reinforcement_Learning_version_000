"""
Sistema de treinamento orquestrado.
"""

from typing import Dict, Any, Optional, Callable, List
import time
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
import json
import yaml
from tqdm import tqdm
import gymnasium as gym

from .problem import ProblemSpec
from .agent import Agent
from .memory import UnifiedMemory, RolloutBuffer


class Trainer:
    """Orquestrador de treinamento."""
    
    def __init__(self,
                 problem: ProblemSpec,
                 agent: Agent,
                 config: Dict[str, Any] = None):
        
        self.problem = problem
        self.agent = agent
        self.config = self._merge_configs(config)
        
        # Cria ambiente
        self.env = self._create_env()
        self.eval_env = self._create_env()
        
        # Memória
        self.memory = self._create_memory()
        
        # Logging
        self.logger = Logger(self.config['log_dir'])
        
        # Estado do treinamento
        self.total_steps = 0
        self.total_episodes = 0
        self.best_reward = -float('inf')
        
        # Callbacks
        self.callbacks = []
    
    def _merge_configs(self, user_config: Optional[Dict]) -> Dict:
        """Mescla configuração do usuário com defaults."""
        defaults = {
            # Treinamento
            'total_steps': 1_000_000,
            'batch_size': 32,
            'learning_rate': 3e-4,
            'gamma': 0.99,
            'seed': 0,
            
            # Ambiente
            'n_envs': 1,  # Vetorização
            'max_episode_steps': 1000,
            
            # Avaliação
            'eval_frequency': 10000,
            'n_eval_episodes': 10,
            
            # Logging
            'log_dir': f'./logs/{datetime.now():%Y%m%d_%H%M%S}',
            'save_frequency': 50000,
            'verbose': 1,
            
            # Early stopping
            'target_reward': None,
            'patience': 50,
            
            # Dispositivo
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        
        if user_config:
            defaults.update(user_config)
        
        return defaults
    
    def _create_env(self) -> gym.Env:
        """Cria ambiente baseado no problema."""
        # Por enquanto, retorna um wrapper simples
        # Na versão completa, isso seria um ambiente Gymnasium completo
        class ProblemEnv(gym.Env):
            def __init__(self, problem_spec):
                self.problem = problem_spec
                self.observation_space = problem_spec.observation_space
                self.action_space = problem_spec.action_space
                self.state = None
            
            def reset(self, seed=None):
                if seed:
                    np.random.seed(seed)
                # Inicializa estado
                self.state = self._sample_initial_state()
                return self.state, {}
            
            def step(self, action):
                # Simula transição
                next_state = self._transition(self.state, action)
                
                # Calcula recompensas
                rewards = self.problem.compute_reward(
                    self.state, action, next_state
                )
                reward = self.problem.aggregate_rewards(rewards)
                
                # Verifica término
                done = self._is_terminal(next_state)
                truncated = False
                
                self.state = next_state
                
                return next_state, reward, done, truncated, {'rewards': rewards}
            
            def _sample_initial_state(self):
                """Amostra estado inicial."""
                state = {}
                for name, spec in self.problem.observation_spec.items():
                    if spec['type'] == 'continuous':
                        state[name] = np.random.uniform(
                            spec['low'], spec['high']
                        )
                    else:
                        state[name] = np.random.randint(0, spec['n'])
                return state
            
            def _transition(self, state, action):
                """Simula transição de estado."""
                # Placeholder - na prática seria a dinâmica real
                next_state = state.copy()
                # Adiciona pequena perturbação
                for key in next_state:
                    if isinstance(next_state[key], (int, float)):
                        next_state[key] += np.random.normal(0, 0.1)
                return next_state
            
            def _is_terminal(self, state):
                """Verifica condições de término."""
                # Verifica violações de restrições hard
                for constraint in self.problem.constraints:
                    if constraint.hard and not constraint.is_satisfied(state):
                        return True
                return False
        
        return ProblemEnv(self.problem)
    
    def _create_memory(self) -> UnifiedMemory:
        """Cria sistema de memória apropriado."""
        memory_config = {
            'capacity': self.config.get('memory_capacity', 10000),
            'batch_size': self.config['batch_size'],
            'prioritized': self.config.get('prioritized_replay', False),
            'gamma': self.config['gamma']
        }
        
        return UnifiedMemory(**memory_config)
    
    def fit(self, 
            total_steps: Optional[int] = None,
            target_reward: Optional[float] = None,
            callbacks: List[Callable] = None):
        """Executa loop de treinamento principal."""
        
        total_steps = total_steps or self.config['total_steps']
        target_reward = target_reward or self.config['target_reward']
        
        if callbacks:
            self.callbacks.extend(callbacks)
        
        # Setup inicial
        self._setup_training()
        
        # Progress bar
        pbar = tqdm(total=total_steps, desc="Training")
        
        # Loop principal
        obs, _ = self.env.reset(seed=self.config['seed'])
        episode_reward = 0
        episode_steps = 0
        
        while self.total_steps < total_steps:
            # Coleta experiência
            decision = self.agent.act(obs, deterministic=False)
            
            next_obs, reward, done, truncated, info = self.env.step(
                decision.action
            )
            
            # Armazena na memória
            self.memory.add(
                state=obs,
                action=decision.action,
                reward=reward,
                next_state=next_obs,
                done=done or truncated,
                log_prob=decision.log_prob,
                value=decision.value,
                reward_dict=info.get('rewards', {})
            )
            
            episode_reward += reward
            episode_steps += 1
            self.total_steps += 1
            
            # Atualiza agente
            if self.memory.is_ready() and self.total_steps % self.config.get('update_frequency', 1) == 0:
                metrics = self._update_agent()
                self.logger.log_metrics(metrics, self.total_steps)
            
            # Fim do episódio
            if done or truncated:
                self.total_episodes += 1
                
                # Log episódio
                self.logger.log_episode({
                    'reward': episode_reward,
                    'length': episode_steps,
                    'episode': self.total_episodes
                })
                
                # Reset
                obs, _ = self.env.reset()
                episode_reward = 0
                episode_steps = 0
                
                # Callbacks
                for callback in self.callbacks:
                    callback(self)
            else:
                obs = next_obs
            
            # Avaliação periódica
            if self.total_steps % self.config['eval_frequency'] == 0:
                eval_reward = self.evaluate()
                
                # Salva melhor modelo
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    self.save_checkpoint('best_model.pt')
                
                # Early stopping
                if target_reward and eval_reward >= target_reward:
                    print(f"\nTarget reward {target_reward} achieved!")
                    break
            
            # Checkpoint periódico
            if self.total_steps % self.config['save_frequency'] == 0:
                self.save_checkpoint(f'checkpoint_{self.total_steps}.pt')
            
            # Atualiza progress bar
            pbar.update(1)
            pbar.set_postfix({
                'ep_reward': episode_reward,
                'episodes': self.total_episodes
            })
        
        pbar.close()
        
        # Salva modelo final
        self.save_checkpoint('final_model.pt')
        
        return self.agent
    
    def _setup_training(self):
        """Configuração inicial do treinamento."""
        # Set seeds
        np.random.seed(self.config['seed'])
        torch.manual_seed(self.config['seed'])
        
        # Move agent para dispositivo
        self.agent.to(self.config['device'])
        
        # Cria diretórios
        Path(self.config['log_dir']).mkdir(parents=True, exist_ok=True)
        
        # Salva configuração
        config_path = Path(self.config['log_dir']) / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f)
    
    def _update_agent(self) -> Dict[str, float]:
        """Atualiza o agente com dados da memória."""
        # Amostra batch
        batch = self.memory.sample(method='uniform')
        
        # Move para dispositivo
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.config['device'])
        
        # Atualiza agente
        metrics = self.agent.update(batch)
        
        # Atualiza prioridades se usar PER
        if self.config.get('prioritized_replay') and 'td_errors' in metrics:
            self.memory.update_priorities(
                batch['indices'],
                metrics['td_errors'].cpu().numpy()
            )
        
        return metrics
    
    def evaluate(self, n_episodes: Optional[int] = None) -> float:
        """Avalia o agente."""
        n_episodes = n_episodes or self.config['n_eval_episodes']
        
        self.agent.eval_mode()
        
        episode_rewards = []
        episode_lengths = []
        
        for _ in range(n_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                # Ação determinística na avaliação
                decision = self.agent.act(obs, deterministic=True)
                obs, reward, done, truncated, _ = self.eval_env.step(
                    decision.action
                )
                
                episode_reward += reward
                episode_length += 1
                done = done or truncated
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        self.agent.train_mode()
        
        # Log resultados
        eval_metrics = {
            'eval_reward_mean': np.mean(episode_rewards),
            'eval_reward_std': np.std(episode_rewards),
            'eval_length_mean': np.mean(episode_lengths)
        }
        
        self.logger.log_metrics(eval_metrics, self.total_steps)
        
        if self.config['verbose']:
            print(f"\nEval at {self.total_steps} steps:")
            print(f"  Reward: {eval_metrics['eval_reward_mean']:.2f} ± "
                  f"{eval_metrics['eval_reward_std']:.2f}")
        
        return eval_metrics['eval_reward_mean']
    
    def save_checkpoint(self, filename: str):
        """Salva checkpoint do treinamento."""
        checkpoint_path = Path(self.config['log_dir']) / filename
        
        checkpoint = {
            'agent_state': self.agent.state_dict() if hasattr(self.agent, 'state_dict') else None,
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
            'best_reward': self.best_reward,
            'config': self.config
        }
        
        # Salva modelo do agente
        self.agent.save(checkpoint_path.with_suffix('.agent'))
        
        # Salva checkpoint geral
        torch.save(checkpoint, checkpoint_path)
    
    def load_checkpoint(self, path: str):
        """Carrega checkpoint."""
        checkpoint = torch.load(path)
        
        self.total_steps = checkpoint['total_steps']
        self.total_episodes = checkpoint['total_episodes']
        self.best_reward = checkpoint['best_reward']
        
        # Carrega agente
        agent_path = Path(path).with_suffix('.agent')
        if agent_path.exists():
            self.agent.load(agent_path)


class Logger:
    """Sistema de logging."""
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Arquivos de log
        self.metrics_file = self.log_dir / 'metrics.jsonl'
        self.episodes_file = self.log_dir / 'episodes.jsonl'
        
        # Buffer para métricas
        self.metrics_buffer = []
        self.episodes_buffer = []
    
    def log_metrics(self, metrics: Dict[str, Any], step: int):
        """Registra métricas."""
        entry = {'step': step, 'timestamp': time.time(), **metrics}
        self.metrics_buffer.append(entry)
        
        # Flush periodicamente
        if len(self.metrics_buffer) >= 100:
            self._flush_metrics()
    
    def log_episode(self, episode_data: Dict[str, Any]):
        """Registra dados do episódio."""
        entry = {'timestamp': time.time(), **episode_data}
        self.episodes_buffer.append(entry)
        
        if len(self.episodes_buffer) >= 10:
            self._flush_episodes()
    
    def _flush_metrics(self):
        """Salva métricas em disco."""
        with open(self.metrics_file, 'a') as f:
            for entry in self.metrics_buffer:
                f.write(json.dumps(entry) + '\n')
        self.metrics_buffer = []
    
    def _flush_episodes(self):
        """Salva episódios em disco."""
        with open(self.episodes_file, 'a') as f:
            for entry in self.episodes_buffer:
                f.write(json.dumps(entry) + '\n')
        self.episodes_buffer = []
    
    def close(self):
        """Finaliza logging."""
        self._flush_metrics()
        self._flush_episodes()


def train(problem: ProblemSpec,
          algorithm: str = "auto",
          hours: float = None,
          target_score: float = None,
          config: Dict = None) -> Agent:
    """Função de conveniência para treinamento rápido."""

    from ..algorithms.registry import AlgorithmRegistry
    
    # Seleciona algoritmo
    if algorithm == "auto":
        algorithm = AlgorithmRegistry.auto_select(problem)
    
    # Cria agente
    agent_class = AlgorithmRegistry.get(algorithm)
    agent = agent_class(problem, config)
    
    # Configura tempo limite
    if hours:
        total_steps = int(hours * 3600 * 1000)  # Estimativa de steps/segundo
    else:
        total_steps = config.get('total_steps', 1_000_000) if config else 1_000_000
    
    # Cria trainer
    trainer_config = config or {}
    trainer_config['total_steps'] = total_steps
    trainer_config['target_reward'] = target_score
    
    trainer = Trainer(problem, agent, trainer_config)
    
    # Treina
    return trainer.fit()