"""
Exemplo Avançado 7: Ensemble Learning & Model Selection

Este exemplo demonstra:
- Ensemble de múltiplos agentes RL
- Meta-learning: aprender qual agente usar
- Combinação de predições (voting, stacking)
- Model selection dinâmico
- Robustez via diversidade

Técnicas:
1. Voting Ensemble: voto majoritário/ponderado
2. Stacking: meta-modelo aprende a combinar
3. Dynamic Selection: escolhe melhor modelo por contexto
4. Mixture of Experts: combina especializações
"""

import business_rl as brl
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict


# ===== PROBLEMA BASE: CUSTOMER CHURN PREDICTION & ACTION =====
@brl.problem(name="ChurnPrevention")
class ChurnPrevention:
    """
    Problema: Prever e prevenir churn de clientes

    Ação: Qual intervenção fazer com cada cliente
    """

    obs = brl.Dict(
        # Perfil do cliente
        tenure_meses=brl.Box(0, 120),
        valor_mensal=brl.Box(0, 500),
        total_gasto=brl.Box(0, 20000),

        # Comportamento
        uso_ultimos_30d=brl.Box(0, 1),  # Normalizado
        frequencia_login=brl.Box(0, 1),
        features_usadas=brl.Box(0, 1),

        # Engajamento
        tickets_suporte=brl.Box(0, 20),
        nps_score=brl.Box(0, 10),
        tempo_desde_ultimo_uso=brl.Box(0, 90),

        # Histórico financeiro
        atrasos_pagamento=brl.Box(0, 10),
        mudancas_plano=brl.Box(0, 5),

        # Predições de modelos externos
        propensao_churn_ml=brl.Box(0, 1),  # Modelo ML tradicional
        segmento_cliente=brl.Discrete(5),   # 0-4

        # Contexto temporal
        mes_ano=brl.Discrete(12),
        dias_ate_renovacao=brl.Box(0, 365)
    )

    action = brl.Discrete(
        5,
        labels=[
            "sem_acao",         # Não fazer nada
            "desconto_10",      # 10% desconto
            "upgrade_gratis",   # Upgrade temporário
            "contato_vendas",   # Ligar proativamente
            "oferta_premium"    # Oferta especial
        ]
    )

    objectives = brl.Terms(
        retencao=0.50,      # Evitar churn
        custo=0.25,         # Minimizar custo da ação
        ltv=0.25           # Maximizar lifetime value
    )

    def reward_retencao(self, state, action, next_state):
        """Recompensa por reter cliente."""
        propensao = state['propensao_churn_ml']

        # Sem ação em cliente de alto risco = penalidade
        if propensao > 0.7 and action == 0:
            return -50

        # Ação em cliente de alto risco = recompensa
        if propensao > 0.7 and action > 0:
            return 100 * (1 - propensao)

        # Ação desnecessária em cliente de baixo risco
        if propensao < 0.3 and action > 0:
            return -20

        return 0

    def reward_custo(self, state, action, next_state):
        """Penaliza custos das ações."""
        custos = {
            0: 0,      # Sem ação
            1: -30,    # Desconto
            2: -50,    # Upgrade
            3: -20,    # Contato
            4: -100    # Premium
        }
        return custos[action]

    def reward_ltv(self, state, action, next_state):
        """Considera lifetime value."""
        ltv_estimado = state['valor_mensal'] * state['tenure_meses'] * 0.5

        # Ações mais agressivas para clientes de alto valor
        if ltv_estimado > 5000:
            if action in [2, 4]:  # Upgrade ou Premium
                return 30
        return 0


class EnsembleChurnSystem:
    """
    Sistema ensemble para prevenção de churn.

    Combina múltiplos agentes especializados.
    """

    def __init__(self):
        # Agentes especializados
        self.agents = {}

        # Meta-learner
        self.meta_model = None

        # Histórico de performance
        self.performance_history = defaultdict(list)

        # Pesos para voting
        self.voting_weights = None

    def treinar_ensemble(self):
        """Treina múltiplos agentes com diferentes configurações."""
        print("=" * 70)
        print("TREINAMENTO: ENSEMBLE DE AGENTES")
        print("=" * 70)

        problema = ChurnPrevention()

        # Configurações diferentes para diversidade
        configs = {
            'conservative': {
                'name': 'Conservador',
                'config': {
                    'learning_rate': 1e-4,
                    'gamma': 0.99,
                    'ent_coef': 0.001,  # Pouca exploração
                    'clip_range': 0.1
                }
            },
            'aggressive': {
                'name': 'Agressivo',
                'config': {
                    'learning_rate': 5e-4,
                    'gamma': 0.95,
                    'ent_coef': 0.1,    # Muita exploração
                    'clip_range': 0.3
                }
            },
            'balanced': {
                'name': 'Balanceado',
                'config': {
                    'learning_rate': 3e-4,
                    'gamma': 0.97,
                    'ent_coef': 0.01,
                    'clip_range': 0.2
                }
            },
            'long_horizon': {
                'name': 'Longo Prazo',
                'config': {
                    'learning_rate': 2e-4,
                    'gamma': 0.995,     # Mais peso no futuro
                    'ent_coef': 0.02,
                    'clip_range': 0.2
                }
            }
        }

        # Treina cada agente
        for key, setup in configs.items():
            print(f"\n🤖 Treinando agente: {setup['name']}")

            modelo = brl.train(
                problema,
                algorithm='PPO',
                hours=0.3,  # 18 minutos cada
                config=setup['config']
            )

            self.agents[key] = modelo
            modelo.save(f'./modelos/agent_{key}.pt')

        print("\n✅ Ensemble completo!")

        # Treina meta-learner
        self._treinar_meta_learner()

    def _treinar_meta_learner(self):
        """Treina meta-modelo para combinar agentes."""
        print("\n🧠 Treinando Meta-Learner...")

        @brl.problem(name="MetaLearner")
        class MetaLearner:
            """Aprende qual agente usar em cada contexto."""

            obs = brl.Dict(
                # Contexto do problema
                propensao_churn=brl.Box(0, 1),
                valor_cliente=brl.Box(0, 1),
                tenure=brl.Box(0, 1),

                # Performance histórica de cada agente
                acc_conservative=brl.Box(0, 1),
                acc_aggressive=brl.Box(0, 1),
                acc_balanced=brl.Box(0, 1),
                acc_long_horizon=brl.Box(0, 1),

                # Predições de cada agente (one-hot das 5 ações)
                pred_conservative=brl.Discrete(5),
                pred_aggressive=brl.Discrete(5),
                pred_balanced=brl.Discrete(5),
                pred_long_horizon=brl.Discrete(5),

                # Concordância entre agentes
                consensus_level=brl.Box(0, 1)
            )

            # Escolhe qual agente usar
            action = brl.Discrete(
                4,
                labels=["conservative", "aggressive", "balanced", "long_horizon"]
            )

            objectives = brl.Terms(
                accuracy=0.70,
                diversity=0.30
            )

            def reward_accuracy(self, state, action, next_state):
                """Escolhe agente com melhor histórico."""
                accuracies = [
                    state['acc_conservative'],
                    state['acc_aggressive'],
                    state['acc_balanced'],
                    state['acc_long_horizon']
                ]
                return accuracies[action] * 100

            def reward_diversity(self, state, action, next_state):
                """Incentiva usar diferentes agentes."""
                # Penaliza sempre usar o mesmo
                return np.random.rand() * 10

        problema_meta = MetaLearner()
        self.meta_model = brl.train(
            problema_meta,
            algorithm='PPO',
            hours=0.2,
            config={'learning_rate': 3e-4}
        )

        self.meta_model.save('./modelos/meta_learner.pt')
        print("✅ Meta-learner treinado!")

    def carregar_ensemble(self):
        """Carrega ensemble treinado."""
        print("📦 Carregando ensemble...")

        agent_keys = ['conservative', 'aggressive', 'balanced', 'long_horizon']
        for key in agent_keys:
            self.agents[key] = brl.load(f'./modelos/agent_{key}.pt')

        self.meta_model = brl.load('./modelos/meta_learner.pt')
        print("✅ Ensemble carregado!")

    def predict_voting(self, state: Dict, weights: str = 'uniform') -> int:
        """
        Voting ensemble: combina votos dos agentes.

        Args:
            weights: 'uniform' ou 'performance'
        """
        # Coleta predições
        predictions = {}
        for key, agent in self.agents.items():
            decisao = agent.decide(state, deterministic=True)
            predictions[key] = decisao.action

        # Voto
        if weights == 'uniform':
            # Voto majoritário simples
            votes = list(predictions.values())
            return max(set(votes), key=votes.count)

        elif weights == 'performance':
            # Voto ponderado por performance histórica
            if self.voting_weights is None:
                # Pesos iguais se não temos histórico
                return self.predict_voting(state, 'uniform')

            weighted_votes = defaultdict(float)
            for key, action in predictions.items():
                weighted_votes[action] += self.voting_weights.get(key, 0.25)

            return max(weighted_votes.items(), key=lambda x: x[1])[0]

    def predict_stacking(self, state: Dict) -> int:
        """
        Stacking: meta-modelo combina predições.
        """
        # Coleta predições dos agentes base
        predictions = {}
        for key, agent in self.agents.items():
            decisao = agent.decide(state, deterministic=True)
            predictions[key] = decisao.action

        # Prepara estado para meta-learner
        meta_state = {
            'propensao_churn': state['propensao_churn_ml'],
            'valor_cliente': state['valor_mensal'] / 500,
            'tenure': state['tenure_meses'] / 120,

            # Performance histórica (simplificado)
            'acc_conservative': 0.85,
            'acc_aggressive': 0.78,
            'acc_balanced': 0.82,
            'acc_long_horizon': 0.80,

            # Predições
            'pred_conservative': predictions['conservative'],
            'pred_aggressive': predictions['aggressive'],
            'pred_balanced': predictions['balanced'],
            'pred_long_horizon': predictions['long_horizon'],

            # Consenso
            'consensus_level': len(set(predictions.values())) / len(predictions)
        }

        # Meta-learner decide qual agente usar
        decisao_meta = self.meta_model.decide(meta_state, deterministic=True)
        agent_escolhido = list(self.agents.keys())[decisao_meta.action]

        # Retorna predição do agente escolhido
        return predictions[agent_escolhido]

    def predict_mixture_of_experts(self, state: Dict) -> int:
        """
        Mixture of Experts: usa agente especializado por contexto.
        """
        propensao = state['propensao_churn_ml']
        valor = state['valor_mensal']
        tenure = state['tenure_meses']

        # Regras de especialização
        if propensao > 0.8:
            # Alto risco: agente agressivo
            agent_key = 'aggressive'

        elif valor > 300 and tenure > 24:
            # Alto valor e longo prazo: conservador
            agent_key = 'conservative'

        elif tenure < 6:
            # Cliente novo: longo prazo
            agent_key = 'long_horizon'

        else:
            # Caso geral: balanceado
            agent_key = 'balanced'

        decisao = self.agents[agent_key].decide(state, deterministic=True)
        return decisao.action


def comparar_estrategias_ensemble():
    """Compara diferentes estratégias de ensemble."""
    print("\n" + "=" * 70)
    print("COMPARAÇÃO: ESTRATÉGIAS DE ENSEMBLE")
    print("=" * 70)

    # Carrega ensemble
    sistema = EnsembleChurnSystem()
    sistema.carregar_ensemble()

    # Gera 100 casos de teste
    n_casos = 100
    resultados = {
        'voting_uniform': {'corretos': 0, 'custo': 0},
        'voting_weighted': {'corretos': 0, 'custo': 0},
        'stacking': {'corretos': 0, 'custo': 0},
        'mixture_experts': {'corretos': 0, 'custo': 0},
        'individual_best': {'corretos': 0, 'custo': 0}
    }

    print(f"\n🧪 Testando com {n_casos} clientes...\n")

    for i in range(n_casos):
        # Gera cliente aleatório
        estado = {
            'tenure_meses': np.random.randint(1, 120),
            'valor_mensal': np.random.rand() * 500,
            'total_gasto': np.random.rand() * 20000,
            'uso_ultimos_30d': np.random.rand(),
            'frequencia_login': np.random.rand(),
            'features_usadas': np.random.rand(),
            'tickets_suporte': np.random.randint(0, 20),
            'nps_score': np.random.randint(0, 11),
            'tempo_desde_ultimo_uso': np.random.randint(0, 90),
            'atrasos_pagamento': np.random.randint(0, 10),
            'mudancas_plano': np.random.randint(0, 5),
            'propensao_churn_ml': np.random.rand(),
            'segmento_cliente': np.random.randint(0, 5),
            'mes_ano': np.random.randint(1, 13),
            'dias_ate_renovacao': np.random.randint(0, 365)
        }

        # Ação ótima (simulada)
        propensao = estado['propensao_churn_ml']
        if propensao > 0.8:
            acao_otima = 4  # Premium
        elif propensao > 0.6:
            acao_otima = 2  # Upgrade
        elif propensao > 0.4:
            acao_otima = 1  # Desconto
        else:
            acao_otima = 0  # Sem ação

        # Testa cada estratégia
        acoes_preditas = {
            'voting_uniform': sistema.predict_voting(estado, 'uniform'),
            'voting_weighted': sistema.predict_voting(estado, 'performance'),
            'stacking': sistema.predict_stacking(estado),
            'mixture_experts': sistema.predict_mixture_of_experts(estado),
        }

        # Adiciona melhor agente individual
        melhor_individual = sistema.agents['balanced'].decide(
            estado, deterministic=True
        ).action
        acoes_preditas['individual_best'] = melhor_individual

        # Custos das ações
        custos_acao = {0: 0, 1: 30, 2: 50, 3: 20, 4: 100}

        # Avalia cada predição
        for estrategia, acao in acoes_preditas.items():
            if acao == acao_otima:
                resultados[estrategia]['corretos'] += 1

            resultados[estrategia]['custo'] += custos_acao[acao]

    # Mostra resultados
    print("\nResultados:")
    print("-" * 70)
    print(f"{'Estratégia':<25} {'Acurácia':<15} {'Custo Total':<15} {'Score'}")
    print("-" * 70)

    for estrategia, res in resultados.items():
        acuracia = res['corretos'] / n_casos
        custo = res['custo']
        score = acuracia * 100 - custo / 100  # Score combinado

        nome_estrategia = {
            'voting_uniform': 'Voting (Uniforme)',
            'voting_weighted': 'Voting (Ponderado)',
            'stacking': 'Stacking',
            'mixture_experts': 'Mixture of Experts',
            'individual_best': 'Melhor Individual'
        }[estrategia]

        print(f"{nome_estrategia:<25} {acuracia:<15.2%} R$ {custo:<12,.2f} {score:.2f}")

    # Melhor estratégia
    melhor = max(resultados.items(),
                key=lambda x: x[1]['corretos'] / n_casos)

    print("\n" + "=" * 70)
    print(f"🏆 Melhor estratégia: {melhor[0].upper()}")
    print(f"   Acurácia: {melhor[1]['corretos']/n_casos:.2%}")
    print("=" * 70)


def demo_ensemble_diversity():
    """Demonstra a importância da diversidade no ensemble."""
    print("\n" + "=" * 70)
    print("ANÁLISE: DIVERSIDADE DO ENSEMBLE")
    print("=" * 70)

    sistema = EnsembleChurnSystem()
    sistema.carregar_ensemble()

    # Testa em 50 casos
    n_casos = 50
    agreement_scores = []

    print(f"\nAnalisando concordância entre agentes em {n_casos} casos...\n")

    for i in range(n_casos):
        estado = {
            'tenure_meses': np.random.randint(1, 120),
            'valor_mensal': np.random.rand() * 500,
            'total_gasto': np.random.rand() * 20000,
            'uso_ultimos_30d': np.random.rand(),
            'frequencia_login': np.random.rand(),
            'features_usadas': np.random.rand(),
            'tickets_suporte': np.random.randint(0, 20),
            'nps_score': np.random.randint(0, 11),
            'tempo_desde_ultimo_uso': np.random.randint(0, 90),
            'atrasos_pagamento': np.random.randint(0, 10),
            'mudancas_plano': np.random.randint(0, 5),
            'propensao_churn_ml': np.random.rand(),
            'segmento_cliente': np.random.randint(0, 5),
            'mes_ano': np.random.randint(1, 13),
            'dias_ate_renovacao': np.random.randint(0, 365)
        }

        # Predições de cada agente
        predictions = []
        for key, agent in sistema.agents.items():
            decisao = agent.decide(estado, deterministic=True)
            predictions.append(decisao.action)

        # Nível de concordância
        unique_preds = len(set(predictions))
        agreement = 1 - (unique_preds - 1) / (len(predictions) - 1)
        agreement_scores.append(agreement)

    avg_agreement = np.mean(agreement_scores)
    diversity = 1 - avg_agreement

    print(f"Concordância média: {avg_agreement:.2%}")
    print(f"Diversidade: {diversity:.2%}")
    print("\nInterpretação:")
    if diversity > 0.4:
        print("  ✅ Alta diversidade - ensemble robusto!")
    elif diversity > 0.2:
        print("  ⚠️  Diversidade moderada")
    else:
        print("  ❌ Baixa diversidade - agentes muito similares")


if __name__ == "__main__":
    import os
    os.makedirs('./modelos', exist_ok=True)

    print("🚀 Iniciando exemplo de Ensemble Learning\n")

    # Treina ensemble
    sistema = EnsembleChurnSystem()
    sistema.treinar_ensemble()

    # Comparações
    comparar_estrategias_ensemble()

    # Análise de diversidade
    demo_ensemble_diversity()

    print("\n" + "=" * 70)
    print("✅ Exemplo concluído!")
    print("=" * 70)
