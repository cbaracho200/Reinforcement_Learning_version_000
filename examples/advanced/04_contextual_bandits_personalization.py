"""
Exemplo Avançado 4: Contextual Bandits para Personalização

Este exemplo demonstra:
- Contextual Multi-Armed Bandits
- Personalização de conteúdo em tempo real
- Exploration vs Exploitation (Thompson Sampling, UCB)
- A/B testing inteligente
- Cold-start problem

Casos de uso:
- Recomendação de produtos
- Personalização de emails
- Otimização de banners/ads
- Seleção de conteúdo
"""

import business_rl as brl
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict


@brl.problem(name="PersonalizacaoContextual")
class PersonalizacaoContextual:
    """
    Problema: Escolher qual conteúdo mostrar para cada usuário

    Contextual Bandit: Temos diferentes "braços" (opções de conteúdo)
    e precisamos escolher qual mostrar baseado no contexto do usuário.
    """

    # ===== OBSERVAÇÕES (CONTEXTO) =====
    obs = brl.Dict(
        # Perfil do usuário
        idade_normalizada=brl.Box(0, 1),  # 18-80 anos normalizado
        genero=brl.Discrete(3, labels=["masculino", "feminino", "outro"]),

        # Comportamento histórico
        tempo_plataforma_meses=brl.Box(0, 1),  # 0-60 meses normalizado
        taxa_engajamento=brl.Box(0, 1),  # 0-100% normalizado
        n_compras_anteriores=brl.Box(0, 1),  # 0-100 normalizado

        # Contexto da sessão
        hora_do_dia=brl.Discrete(24),
        dia_semana=brl.Discrete(7),
        dispositivo=brl.Discrete(3, labels=["mobile", "desktop", "tablet"]),

        # Categoria de interesse (one-hot de 5 categorias)
        interesses=brl.Box(0, 1, shape=(5,)),

        # Histórico de interações com cada tipo de conteúdo
        # [produto, artigo, video, oferta, tutorial]
        historico_cliques=brl.Box(0, 1, shape=(5,)),
        historico_conversoes=brl.Box(0, 1, shape=(5,)),

        # Segmento do usuário (0=novo, 1=ativo, 2=vip, 3=inativo)
        segmento=brl.Discrete(4, labels=["novo", "ativo", "vip", "inativo"]),

        # Score de propensão (calculado por modelo externo)
        propensao_compra=brl.Box(0, 1),
        propensao_churn=brl.Box(0, 1)
    )

    # ===== AÇÕES (BRAÇOS DO BANDIT) =====
    action = brl.Discrete(
        5,
        labels=[
            "recomendar_produto",    # Braço 0
            "mostrar_artigo",        # Braço 1
            "sugerir_video",         # Braço 2
            "exibir_oferta",         # Braço 3
            "oferecer_tutorial"      # Braço 4
        ]
    )

    # ===== OBJETIVOS =====
    objectives = brl.Terms(
        engagement=0.40,        # 40% taxa de clique
        conversion=0.35,        # 35% conversão
        satisfaction=0.15,      # 15% satisfação do usuário
        exploration=0.10        # 10% exploração de novos braços
    )

    # ===== FUNÇÕES DE RECOMPENSA =====

    def reward_engagement(self, state, action, next_state):
        """Recompensa baseada na taxa de clique esperada."""
        # Usa histórico de cliques como proxy
        taxa_clique_historica = state['historico_cliques'][action]

        # Ajusta baseado no segmento
        segmento = state['segmento']
        multiplicador_segmento = {
            0: 0.8,   # Novo: menos engajado
            1: 1.0,   # Ativo: baseline
            2: 1.3,   # VIP: mais engajado
            3: 0.5    # Inativo: muito menos engajado
        }[segmento]

        # Ajusta baseado no horário
        hora = state['hora_do_dia']
        multiplicador_hora = 1.2 if 18 <= hora <= 22 else 1.0  # Pico à noite

        recompensa = (taxa_clique_historica *
                     multiplicador_segmento *
                     multiplicador_hora)

        return recompensa * 100

    def reward_conversion(self, state, action, next_state):
        """Recompensa baseada na probabilidade de conversão."""
        # Taxa de conversão histórica para este tipo de conteúdo
        taxa_conv_historica = state['historico_conversoes'][action]

        # Ajusta pela propensão de compra
        propensao = state['propensao_compra']

        # Ofertas convertem melhor com alta propensão
        if action == 3:  # Oferta
            bonus_oferta = propensao * 0.5
        else:
            bonus_oferta = 0

        recompensa = (taxa_conv_historica + bonus_oferta) * propensao

        return recompensa * 100

    def reward_satisfaction(self, state, action, next_state):
        """Penaliza mostrar conteúdo repetitivo ou inadequado."""
        # Penaliza se já teve muitas interações deste tipo
        frequencia = state['historico_cliques'][action]

        if frequencia > 0.8:  # Muito repetitivo
            penalidade = -20
        elif frequencia > 0.5:
            penalidade = -5
        else:
            penalidade = 0

        # Recompensa por match com interesses
        match_interesse = state['interesses'][action % 5]
        bonus_interesse = match_interesse * 10

        return penalidade + bonus_interesse

    def reward_exploration(self, state, action, next_state):
        """Incentiva explorar braços menos testados."""
        # Menos cliques = mais exploração
        exploracao = 1 - state['historico_cliques'][action]

        return exploracao * 10


class ContextualBanditEnsemble:
    """
    Ensemble de estratégias de Contextual Bandits:
    - Thompson Sampling
    - Upper Confidence Bound (UCB)
    - Epsilon-Greedy
    - RL Agent (treinado)
    """

    def __init__(self, n_arms: int = 5):
        self.n_arms = n_arms

        # Thompson Sampling: parâmetros Beta para cada braço
        self.alpha = np.ones(n_arms)  # Sucessos
        self.beta = np.ones(n_arms)   # Falhas

        # UCB: contadores
        self.arm_counts = np.zeros(n_arms)
        self.arm_rewards = np.zeros(n_arms)
        self.total_count = 0

        # Epsilon-Greedy
        self.epsilon = 0.1
        self.q_values = np.zeros(n_arms)
        self.n_pulls = np.zeros(n_arms)

        # RL Agent
        self.rl_model = None

        # Histórico de performance
        self.strategy_performance = {
            'thompson': [],
            'ucb': [],
            'epsilon_greedy': [],
            'rl_agent': []
        }

    def thompson_sampling(self, context: Dict) -> int:
        """Thompson Sampling: amostra da distribuição Beta."""
        samples = np.random.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def ucb(self, context: Dict) -> int:
        """Upper Confidence Bound."""
        if self.total_count < self.n_arms:
            # Garantir que todos os braços foram testados
            return int(self.total_count)

        # UCB formula
        ucb_values = (self.arm_rewards / (self.arm_counts + 1e-6) +
                     np.sqrt(2 * np.log(self.total_count + 1) /
                            (self.arm_counts + 1e-6)))

        return int(np.argmax(ucb_values))

    def epsilon_greedy(self, context: Dict) -> int:
        """Epsilon-Greedy: explora com probabilidade epsilon."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)
        else:
            return int(np.argmax(self.q_values))

    def rl_agent(self, context: Dict) -> int:
        """RL Agent treinado."""
        if self.rl_model is None:
            return np.random.randint(self.n_arms)

        decisao = self.rl_model.decide(context, deterministic=True)
        return decisao.action

    def select_arm(self, context: Dict, strategy: str = 'thompson') -> int:
        """Seleciona um braço usando a estratégia especificada."""
        strategies = {
            'thompson': self.thompson_sampling,
            'ucb': self.ucb,
            'epsilon_greedy': self.epsilon_greedy,
            'rl_agent': self.rl_agent
        }

        return strategies[strategy](context)

    def update(self, arm: int, reward: float):
        """Atualiza estatísticas após observar recompensa."""
        # Thompson Sampling
        if reward > 0.5:  # Considera sucesso se reward > 0.5
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1

        # UCB
        self.arm_counts[arm] += 1
        self.arm_rewards[arm] += reward
        self.total_count += 1

        # Epsilon-Greedy
        self.n_pulls[arm] += 1
        self.q_values[arm] += (reward - self.q_values[arm]) / self.n_pulls[arm]

    def set_rl_model(self, model):
        """Define o modelo RL treinado."""
        self.rl_model = model


def treinar_modelo_bandit():
    """Treina o modelo de personalização contextual."""
    print("=" * 70)
    print("TREINAMENTO: CONTEXTUAL BANDIT COM RL")
    print("=" * 70)

    problema = PersonalizacaoContextual()

    print("\n📊 Informações do problema:")
    print(problema.get_info())

    print("\n🏋️ Iniciando treino (1.5 horas)...")
    modelo = brl.train(
        problema,
        algorithm='PPO',
        hours=1.5,
        config={
            'learning_rate': 3e-4,
            'batch_size': 256,
            'n_epochs': 10,
            'gamma': 0.95,  # Horizonte mais curto (bandit)
            'ent_coef': 0.05  # Mais exploração
        }
    )

    modelo.save('./modelos/contextual_bandit.pt')
    print("\n✅ Modelo salvo!")

    return modelo


def comparar_estrategias():
    """Compara diferentes estratégias de Contextual Bandit."""
    print("\n" + "=" * 70)
    print("COMPARAÇÃO: ESTRATÉGIAS DE BANDIT")
    print("=" * 70)

    # Carrega modelo RL
    modelo_rl = brl.load('./modelos/contextual_bandit.pt')

    # Cria ensemble
    ensemble = ContextualBanditEnsemble(n_arms=5)
    ensemble.set_rl_model(modelo_rl)

    # Simula 1000 usuários
    n_usuarios = 1000
    estrategias = ['thompson', 'ucb', 'epsilon_greedy', 'rl_agent']

    resultados = {s: {'cliques': 0, 'conversoes': 0} for s in estrategias}

    print("\n🧪 Simulando 1000 usuários...\n")

    for i in range(n_usuarios):
        # Gera contexto aleatório de usuário
        contexto = {
            'idade_normalizada': np.random.rand(),
            'genero': np.random.randint(3),
            'tempo_plataforma_meses': np.random.rand(),
            'taxa_engajamento': np.random.rand(),
            'n_compras_anteriores': np.random.rand(),
            'hora_do_dia': np.random.randint(24),
            'dia_semana': np.random.randint(7),
            'dispositivo': np.random.randint(3),
            'interesses': np.random.rand(5),
            'historico_cliques': np.random.rand(5) * 0.5,
            'historico_conversoes': np.random.rand(5) * 0.3,
            'segmento': np.random.randint(4),
            'propensao_compra': np.random.rand(),
            'propensao_churn': np.random.rand()
        }

        # Testa cada estratégia
        for estrategia in estrategias:
            # Seleciona braço
            braco = ensemble.select_arm(contexto, estrategia)

            # Simula resultado (probabilidade baseada no contexto)
            prob_clique = 0.3 + contexto['taxa_engajamento'] * 0.4
            prob_conversao = 0.1 + contexto['propensao_compra'] * 0.3

            # Ajusta por tipo de conteúdo
            if braco == 3:  # Oferta
                prob_clique *= 1.2
                prob_conversao *= 1.5

            clicou = np.random.rand() < prob_clique
            converteu = clicou and (np.random.rand() < prob_conversao)

            if clicou:
                resultados[estrategia]['cliques'] += 1
            if converteu:
                resultados[estrategia]['conversoes'] += 1

            # Atualiza apenas para estratégias clássicas
            if estrategia != 'rl_agent':
                reward = 1.0 if converteu else (0.5 if clicou else 0.0)
                ensemble.update(braco, reward)

    # Mostra resultados
    print("Resultados após 1000 usuários:")
    print("-" * 70)
    print(f"{'Estratégia':<20} {'Taxa Clique':<15} {'Taxa Conversão':<15} {'Score'}")
    print("-" * 70)

    for estrategia in estrategias:
        taxa_clique = resultados[estrategia]['cliques'] / n_usuarios
        taxa_conversao = resultados[estrategia]['conversoes'] / n_usuarios
        score = taxa_clique * 0.4 + taxa_conversao * 0.6  # Score ponderado

        nome_estrategia = {
            'thompson': 'Thompson Sampling',
            'ucb': 'UCB',
            'epsilon_greedy': 'Epsilon-Greedy',
            'rl_agent': 'RL Agent (PPO)'
        }[estrategia]

        print(f"{nome_estrategia:<20} {taxa_clique:<15.2%} {taxa_conversao:<15.2%} {score:.4f}")

    # Identifica melhor estratégia
    melhor = max(estrategias,
                key=lambda s: resultados[s]['conversoes'])

    print("\n" + "=" * 70)
    print(f"🏆 Melhor estratégia: {melhor.upper()}")
    print("=" * 70)


def teste_ab_inteligente():
    """Demonstra como usar bandits para A/B testing inteligente."""
    print("\n" + "=" * 70)
    print("A/B TESTING INTELIGENTE COM BANDITS")
    print("=" * 70)

    print("""
Cenário: Teste de 3 variações de email marketing
- Variação A: Email promocional direto
- Variação B: Email educacional + soft sell
- Variação C: Email personalizado baseado em comportamento

Em vez de dividir 33%-33%-33% (A/B/C tradicional),
usamos Thompson Sampling para alocar mais tráfego
para variações que performam melhor.
""")

    # Simula 500 envios
    n_envios = 500
    ensemble = ContextualBanditEnsemble(n_arms=3)

    # Taxa de conversão real (desconhecida inicialmente)
    taxas_reais = [0.05, 0.08, 0.12]  # C é a melhor

    historico_escolhas = []
    historico_conversoes = []

    for i in range(n_envios):
        # Thompson Sampling escolhe variação
        variacao = ensemble.thompson_sampling({})

        # Simula resultado
        converteu = np.random.rand() < taxas_reais[variacao]

        # Atualiza
        ensemble.update(variacao, 1.0 if converteu else 0.0)

        historico_escolhas.append(variacao)
        historico_conversoes.append(converteu)

    # Resultados
    print("\nResultados do teste:")
    print("-" * 70)

    for i in range(3):
        n_enviados = historico_escolhas.count(i)
        if n_enviados > 0:
            conversoes = sum([1 for j, v in enumerate(historico_escolhas)
                            if v == i and historico_conversoes[j]])
            taxa_obs = conversoes / n_enviados

            variacao_nome = ['A', 'B', 'C'][i]
            print(f"Variação {variacao_nome}:")
            print(f"  Enviados: {n_enviados} ({n_enviados/n_envios*100:.1f}%)")
            print(f"  Conversões: {conversoes}")
            print(f"  Taxa observada: {taxa_obs:.2%}")
            print(f"  Taxa real: {taxas_reais[i]:.2%}")
            print()

    # Conversões totais vs A/B/C tradicional
    conversoes_totais = sum(historico_conversoes)
    conversoes_tradicional = n_envios * np.mean(taxas_reais)

    ganho = ((conversoes_totais - conversoes_tradicional) /
             conversoes_tradicional * 100)

    print(f"Conversões com Bandit: {conversoes_totais}")
    print(f"Conversões com A/B/C tradicional: {conversoes_tradicional:.0f}")
    print(f"Ganho: {ganho:+.1f}%")


if __name__ == "__main__":
    import os
    os.makedirs('./modelos', exist_ok=True)

    print("🚀 Iniciando exemplo de Contextual Bandits\n")

    # 1. Treina modelo RL
    modelo = treinar_modelo_bandit()

    # 2. Compara estratégias
    comparar_estrategias()

    # 3. Demonstra A/B testing inteligente
    teste_ab_inteligente()

    print("\n" + "=" * 70)
    print("✅ Exemplo concluído!")
    print("=" * 70)
