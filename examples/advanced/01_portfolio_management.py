"""
Exemplo Avançado 1: Gestão de Portfólio de Investimentos

Este exemplo demonstra:
- Múltiplas observações (preços, volatilidade, correlações)
- Ações contínuas (alocação de capital)
- Múltiplos objetivos (retorno vs risco)
- Restrições (limites de alocação)
- Gestão de risco com CVaR
"""

import business_rl as brl
import numpy as np


@brl.problem(name="GestaoPortfolio")
class GestaoPortfolio:
    """
    Problema: Otimizar alocação de capital em 5 ativos diferentes

    Objetivo: Maximizar retorno enquanto minimiza risco
    """

    # ===== OBSERVAÇÕES =====
    obs = brl.Dict(
        # Preços normalizados dos 5 ativos (0-1)
        precos=brl.Box(0, 1, shape=(5,)),

        # Retornos históricos (últimos 30 dias)
        retornos_historicos=brl.Box(-1, 1, shape=(5, 30)),

        # Volatilidade de cada ativo
        volatilidade=brl.Box(0, 1, shape=(5,)),

        # Capital disponível normalizado
        capital_disponivel=brl.Box(0, 1),

        # Alocação atual do portfólio
        alocacao_atual=brl.Box(0, 1, shape=(5,)),

        # Índice de mercado (S&P500 normalizado)
        indice_mercado=brl.Box(0, 1),

        # Dia do mês (1-30)
        dia=brl.Discrete(30)
    )

    # ===== AÇÕES =====
    action = brl.Dict(
        # Nova alocação para cada ativo (% do capital)
        alocacao=brl.Box(0, 1, shape=(5,)),

        # Rebalancear ou manter?
        rebalancear=brl.Discrete(2, labels=["manter", "rebalancear"])
    )

    # ===== OBJETIVOS =====
    objectives = brl.Terms(
        retorno=0.6,        # 60% peso no retorno
        risco=0.25,         # 25% peso na minimização de risco
        custos=0.15         # 15% peso na minimização de custos
    )

    # ===== RESTRIÇÕES =====
    constraints = {
        # A soma das alocações deve ser <= 100%
        'soma_alocacao': brl.Limit(
            func=lambda s, a: np.sum(a['alocacao']),
            max_val=1.0,
            hard=True  # Nunca pode violar
        ),

        # Nenhum ativo pode ter mais de 40% do capital
        'max_por_ativo': brl.Limit(
            func=lambda s, a: np.max(a['alocacao']),
            max_val=0.4,
            hard=True
        ),

        # Diversificação mínima: pelo menos 3 ativos
        'min_ativos': brl.Limit(
            func=lambda s, a: np.sum(a['alocacao'] > 0.05),
            min_val=3,
            hard=False  # Soft constraint
        )
    }

    # ===== GESTÃO DE RISCO =====
    risk = brl.CVaR(
        alpha=0.05,         # Considera 5% piores cenários
        max_drawdown=0.15   # Máxima perda aceitável: 15%
    )

    # ===== FUNÇÕES DE RECOMPENSA =====

    def reward_retorno(self, state, action, next_state):
        """Calcula o retorno esperado do portfólio."""
        # Retorno médio de cada ativo
        retornos_medios = np.mean(state['retornos_historicos'], axis=1)

        # Retorno ponderado pela alocação
        retorno_portfolio = np.sum(action['alocacao'] * retornos_medios)

        # Escala para [0, 100]
        return retorno_portfolio * 100

    def reward_risco(self, state, action, next_state):
        """Penaliza portfólios com alta volatilidade."""
        # Volatilidade ponderada
        volatilidade_portfolio = np.sum(
            action['alocacao'] * state['volatilidade']
        )

        # Retorna negativo (queremos minimizar)
        return -volatilidade_portfolio * 100

    def reward_custos(self, state, action, next_state):
        """Penaliza custos de transação ao rebalancear."""
        if action['rebalancear'] == 0:  # Manter
            return 0

        # Calcula mudança na alocação
        mudanca = np.sum(np.abs(
            action['alocacao'] - state['alocacao_atual']
        ))

        # Custo de transação: 0.1% por mudança
        custo = -mudanca * 0.1 * state['capital_disponivel']

        return custo * 100


def treinar_modelo_portfolio():
    """Treina o modelo de gestão de portfólio."""
    print("=" * 70)
    print("TREINAMENTO: GESTÃO DE PORTFÓLIO")
    print("=" * 70)

    # Cria o problema
    problema = GestaoPortfolio()

    # Mostra informações
    print("\n📊 Informações do problema:")
    print(problema.get_info())

    # Treina o modelo
    print("\n🏋️ Iniciando treino (2 horas)...")
    modelo = brl.train(
        problema,
        algorithm='SAC',  # SAC é melhor para ações contínuas
        hours=2,
        config={
            'learning_rate': 3e-4,
            'buffer_size': 100000,
            'batch_size': 256,
            'tau': 0.005,
            'gamma': 0.99,
            'ent_coef': 'auto'
        }
    )

    # Salva o modelo
    modelo.save('./modelos/portfolio_management.pt')
    print("\n✅ Modelo salvo em: ./modelos/portfolio_management.pt")

    return modelo


def testar_modelo_portfolio():
    """Testa o modelo com diferentes cenários."""
    print("\n" + "=" * 70)
    print("TESTE: CENÁRIOS DE MERCADO")
    print("=" * 70)

    # Carrega o modelo
    modelo = brl.load('./modelos/portfolio_management.pt')

    # Define cenários de teste
    cenarios = [
        {
            'nome': '📈 Mercado em Alta',
            'estado': {
                'precos': np.array([0.8, 0.7, 0.9, 0.6, 0.85]),
                'retornos_historicos': np.random.randn(5, 30) * 0.02 + 0.01,
                'volatilidade': np.array([0.15, 0.20, 0.10, 0.25, 0.12]),
                'capital_disponivel': 1.0,
                'alocacao_atual': np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
                'indice_mercado': 0.85,
                'dia': 15
            }
        },
        {
            'nome': '📉 Mercado em Baixa',
            'estado': {
                'precos': np.array([0.3, 0.4, 0.2, 0.5, 0.35]),
                'retornos_historicos': np.random.randn(5, 30) * 0.03 - 0.015,
                'volatilidade': np.array([0.35, 0.40, 0.30, 0.45, 0.32]),
                'capital_disponivel': 1.0,
                'alocacao_atual': np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
                'indice_mercado': 0.40,
                'dia': 15
            }
        },
        {
            'nome': '⚖️ Mercado Estável',
            'estado': {
                'precos': np.array([0.5, 0.5, 0.5, 0.5, 0.5]),
                'retornos_historicos': np.random.randn(5, 30) * 0.01,
                'volatilidade': np.array([0.10, 0.12, 0.08, 0.15, 0.11]),
                'capital_disponivel': 1.0,
                'alocacao_atual': np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
                'indice_mercado': 0.60,
                'dia': 15
            }
        }
    ]

    # Testa cada cenário
    for cenario in cenarios:
        print(f"\n{cenario['nome']}")
        print("-" * 70)

        # Decisão do modelo
        decisao = modelo.decide(cenario['estado'], deterministic=True)

        # Mostra alocação recomendada
        alocacao = decisao.action['alocacao']
        rebalancear = decisao.action['rebalancear']

        print(f"Ação: {'Rebalancear' if rebalancear == 1 else 'Manter'}")
        print(f"\nAlocação recomendada:")
        ativos = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
        for i, (ativo, pct) in enumerate(zip(ativos, alocacao)):
            print(f"  {ativo}: {pct*100:6.2f}%")

        print(f"\nConfiança: {decisao.confidence:.2%}")
        print(f"Valor esperado: {decisao.value:.4f}")


def comparar_com_baseline():
    """Compara com estratégia simples (equal weight)."""
    print("\n" + "=" * 70)
    print("COMPARAÇÃO: RL vs Equal Weight")
    print("=" * 70)

    modelo = brl.load('./modelos/portfolio_management.pt')

    # Simula 100 dias de trading
    n_dias = 100
    capital_inicial = 100000

    capital_rl = capital_inicial
    capital_baseline = capital_inicial

    # Estratégia baseline: equal weight (20% cada)
    alocacao_baseline = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

    for dia in range(n_dias):
        # Simula estado do mercado
        estado = {
            'precos': np.random.rand(5),
            'retornos_historicos': np.random.randn(5, 30) * 0.02,
            'volatilidade': np.random.rand(5) * 0.3,
            'capital_disponivel': 1.0,
            'alocacao_atual': np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
            'indice_mercado': np.random.rand(),
            'dia': dia % 30
        }

        # Decisão do RL
        decisao_rl = modelo.decide(estado, deterministic=True)

        # Simula retornos (simplificado)
        retornos = np.random.randn(5) * 0.02 + 0.001

        # Atualiza capital
        capital_rl *= (1 + np.sum(decisao_rl.action['alocacao'] * retornos))
        capital_baseline *= (1 + np.sum(alocacao_baseline * retornos))

    # Resultados
    print(f"\nCapital Inicial: R$ {capital_inicial:,.2f}")
    print(f"\nApós {n_dias} dias:")
    print(f"  Modelo RL:     R$ {capital_rl:,.2f} ({(capital_rl/capital_inicial-1)*100:+.2f}%)")
    print(f"  Equal Weight:  R$ {capital_baseline:,.2f} ({(capital_baseline/capital_inicial-1)*100:+.2f}%)")
    print(f"\nDiferença: R$ {capital_rl - capital_baseline:+,.2f}")


if __name__ == "__main__":
    import os

    # Cria pasta para modelos
    os.makedirs('./modelos', exist_ok=True)

    # 1. Treina o modelo
    print("🚀 Iniciando exemplo de Gestão de Portfólio\n")
    modelo = treinar_modelo_portfolio()

    # 2. Testa com cenários
    testar_modelo_portfolio()

    # 3. Compara com baseline
    comparar_com_baseline()

    print("\n" + "=" * 70)
    print("✅ Exemplo concluído!")
    print("=" * 70)
