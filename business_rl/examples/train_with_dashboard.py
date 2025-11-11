"""
Exemplo com dashboard de monitoramento.
"""

import business_rl as brl
from business_rl import TrainingDashboard


def main():
    # Cria problema
    problem = brl.CampanhaAds()
    
    # Cria agente com configuração customizada
    config = {
        'learning_rate': 3e-4,
        'batch_size': 256,
        'total_steps': 100_000,
        'eval_frequency': 5000,
        'target_reward': 1.5
    }
    
    agent = brl.PPOAgent(problem, config)
    
    # Cria dashboard
    dashboard = TrainingDashboard(port=8080)
    dashboard.start()
    
    # Cria trainer
    trainer = brl.Trainer(problem, agent, config)
    
    # Callback para atualizar dashboard
    def update_dashboard(trainer):
        metrics = trainer.agent.get_metrics()
        dashboard.update_metrics(metrics)
        
        # A cada 10 episódios, adiciona ao dashboard
        if trainer.total_episodes % 10 == 0:
            dashboard.add_episode({
                'episode': trainer.total_episodes,
                'reward': trainer.best_reward
            })
    
    # Treina
    trainer.fit(callbacks=[update_dashboard])
    
    print(f"Treinamento completo!")
    print(f"Melhor recompensa: {trainer.best_reward:.2f}")
    print(f"Dashboard disponível em http://localhost:8080")
    
    # Mantém dashboard rodando
    input("Pressione Enter para encerrar...")
    dashboard.stop()


if __name__ == "__main__":
    main()