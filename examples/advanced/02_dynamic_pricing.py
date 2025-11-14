"""
Exemplo Avançado 2: Precificação Dinâmica para E-commerce

Este exemplo demonstra:
- Observações temporais (hora, dia, sazonalidade)
- Ações híbridas (preço + desconto + promoção)
- Elasticidade de preço e demanda
- Competição com concorrentes
- Otimização de margem vs volume
"""

import business_rl as brl
import numpy as np
from datetime import datetime, timedelta


@brl.problem(name="PrecificacaoDinamica")
class PrecificacaoDinamica:
    """
    Problema: Definir preço ótimo para maximizar lucro

    Considera:
    - Demanda elástica ao preço
    - Sazonalidade
    - Competição
    - Gestão de estoque
    """

    # ===== OBSERVAÇÕES =====
    obs = brl.Dict(
        # Preço atual do produto
        preco_atual=brl.Box(50, 500),

        # Custo unitário
        custo_unitario=brl.Box(20, 200),

        # Estoque disponível
        estoque=brl.Box(0, 1000),

        # Demanda nas últimas 24h
        demanda_24h=brl.Box(0, 500),

        # Demanda na última semana (7 dias)
        demanda_semanal=brl.Box(0, 500, shape=(7,)),

        # Preço do concorrente principal
        preco_concorrente=brl.Box(50, 500),

        # Preço de outros 2 concorrentes
        precos_outros=brl.Box(50, 500, shape=(2,)),

        # Taxa de conversão atual (0-1)
        taxa_conversao=brl.Box(0, 1),

        # Hora do dia (0-23)
        hora=brl.Discrete(24),

        # Dia da semana (0=segunda, 6=domingo)
        dia_semana=brl.Discrete(7),

        # Dia do mês (1-31)
        dia_mes=brl.Discrete(31),

        # Temporada (0=normal, 1=alta, 2=baixa)
        temporada=brl.Discrete(3, labels=["normal", "alta", "baixa"]),

        # Há promoção ativa?
        promocao_ativa=brl.Discrete(2, labels=["nao", "sim"]),

        # Número de visualizações nas últimas 24h
        visualizacoes=brl.Box(0, 10000)
    )

    # ===== AÇÕES =====
    action = brl.Dict(
        # Novo preço a cobrar
        preco=brl.Box(50, 500),

        # Percentual de desconto (0-40%)
        desconto=brl.Box(0, 0.40),

        # Ativar promoção relâmpago?
        promocao_relampago=brl.Discrete(2, labels=["nao", "sim"]),

        # Destacar produto na página?
        destaque=brl.Discrete(3, labels=["nenhum", "normal", "premium"]),

        # Oferecer frete grátis?
        frete_gratis=brl.Discrete(2, labels=["nao", "sim"])
    )

    # ===== OBJETIVOS =====
    objectives = brl.Terms(
        lucro=0.50,              # 50% peso no lucro
        volume_vendas=0.20,      # 20% peso no volume
        competitividade=0.15,    # 15% peso em ser competitivo
        gestao_estoque=0.15      # 15% peso na gestão de estoque
    )

    # ===== RESTRIÇÕES =====
    constraints = {
        # Preço não pode ser menor que custo
        'margem_minima': brl.Limit(
            func=lambda s, a: a['preco'] * (1 - a['desconto']) - s['custo_unitario'],
            min_val=5,  # Margem mínima de R$5
            hard=True
        ),

        # Desconto máximo de 40%
        'desconto_maximo': brl.Limit(
            func=lambda s, a: a['desconto'],
            max_val=0.40,
            hard=True
        ),

        # Não pode ser muito mais caro que concorrentes
        'competicao': brl.Limit(
            func=lambda s, a: a['preco'] - s['preco_concorrente'],
            max_val=100,  # Máximo R$100 mais caro
            hard=False    # Soft constraint
        )
    }

    # ===== FUNÇÕES DE RECOMPENSA =====

    def _calcular_demanda(self, state, action):
        """Modelo de demanda baseado em elasticidade de preço."""
        # Preço final
        preco_final = action['preco'] * (1 - action['desconto'])

        # Demanda base
        demanda_base = state['demanda_24h']

        # Elasticidade de preço: -2.0 (típico para e-commerce)
        elasticidade = -2.0

        # Ratio de preço vs concorrente
        ratio_preco = preco_final / (state['preco_concorrente'] + 1e-6)

        # Ajuste de demanda baseado no preço
        ajuste_preco = ratio_preco ** elasticidade

        # Bônus por destaque
        bonus_destaque = {
            0: 1.0,      # Nenhum
            1: 1.15,     # Normal (+15%)
            2: 1.30      # Premium (+30%)
        }[action['destaque']]

        # Bônus por frete grátis
        bonus_frete = 1.20 if action['frete_gratis'] == 1 else 1.0

        # Bônus por promoção relâmpago
        bonus_promocao = 1.50 if action['promocao_relampago'] == 1 else 1.0

        # Calcula demanda estimada
        demanda = (demanda_base * ajuste_preco *
                   bonus_destaque * bonus_frete * bonus_promocao)

        # Limita pelo estoque
        demanda = min(demanda, state['estoque'])

        return max(0, demanda)

    def reward_lucro(self, state, action, next_state):
        """Maximiza o lucro total."""
        # Preço final
        preco_final = action['preco'] * (1 - action['desconto'])

        # Estima demanda
        demanda = self._calcular_demanda(state, action)

        # Receita
        receita = preco_final * demanda

        # Custos
        custo_produto = state['custo_unitario'] * demanda
        custo_frete = 15 * demanda if action['frete_gratis'] == 1 else 0
        custo_destaque = {0: 0, 1: 50, 2: 200}[action['destaque']]
        custo_promocao = 100 if action['promocao_relampago'] == 1 else 0

        custo_total = custo_produto + custo_frete + custo_destaque + custo_promocao

        # Lucro
        lucro = receita - custo_total

        return lucro / 100  # Normaliza

    def reward_volume_vendas(self, state, action, next_state):
        """Incentiva volume de vendas."""
        demanda = self._calcular_demanda(state, action)
        return demanda / 10  # Normaliza

    def reward_competitividade(self, state, action, next_state):
        """Recompensa por ser competitivo."""
        preco_final = action['preco'] * (1 - action['desconto'])

        # Diferença vs concorrente
        diff = preco_final - state['preco_concorrente']

        if diff < -50:
            # Muito mais barato (pode perder margem)
            return -10
        elif diff < 0:
            # Um pouco mais barato (bom!)
            return 20
        elif diff < 50:
            # Similar (ok)
            return 10
        else:
            # Muito mais caro (ruim)
            return -20

    def reward_gestao_estoque(self, state, action, next_state):
        """Penaliza estoque muito alto ou muito baixo."""
        demanda = self._calcular_demanda(state, action)
        estoque_restante = state['estoque'] - demanda

        if estoque_restante < 10:
            # Risco de ruptura
            return -30
        elif estoque_restante < 50:
            # Estoque baixo
            return -10
        elif estoque_restante > 800:
            # Estoque muito alto (capital parado)
            return -15
        else:
            # Nível adequado
            return 0


def treinar_modelo_pricing():
    """Treina o modelo de precificação."""
    print("=" * 70)
    print("TREINAMENTO: PRECIFICAÇÃO DINÂMICA")
    print("=" * 70)

    problema = PrecificacaoDinamica()

    print("\n📊 Informações do problema:")
    print(problema.get_info())

    print("\n🏋️ Iniciando treino (2 horas)...")
    modelo = brl.train(
        problema,
        algorithm='PPO',  # PPO funciona bem para ações mistas
        hours=2,
        config={
            'learning_rate': 3e-4,
            'batch_size': 256,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01
        }
    )

    modelo.save('./modelos/dynamic_pricing.pt')
    print("\n✅ Modelo salvo!")

    return modelo


def testar_modelo_pricing():
    """Testa o modelo com diferentes cenários."""
    print("\n" + "=" * 70)
    print("TESTE: CENÁRIOS DE PRECIFICAÇÃO")
    print("=" * 70)

    modelo = brl.load('./modelos/dynamic_pricing.pt')

    cenarios = [
        {
            'nome': '🔥 Alta Demanda + Estoque Baixo',
            'estado': {
                'preco_atual': 200,
                'custo_unitario': 80,
                'estoque': 50,
                'demanda_24h': 100,
                'demanda_semanal': np.array([80, 90, 95, 100, 110, 95, 100]),
                'preco_concorrente': 220,
                'precos_outros': np.array([210, 230]),
                'taxa_conversao': 0.08,
                'hora': 14,
                'dia_semana': 2,
                'dia_mes': 15,
                'temporada': 1,  # Alta
                'promocao_ativa': 0,
                'visualizacoes': 5000
            }
        },
        {
            'nome': '📉 Baixa Demanda + Estoque Alto',
            'estado': {
                'preco_atual': 200,
                'custo_unitario': 80,
                'estoque': 900,
                'demanda_24h': 20,
                'demanda_semanal': np.array([25, 22, 20, 18, 20, 22, 19]),
                'preco_concorrente': 180,
                'precos_outros': np.array([175, 185]),
                'taxa_conversao': 0.02,
                'hora': 10,
                'dia_semana': 0,
                'dia_mes': 5,
                'temporada': 2,  # Baixa
                'promocao_ativa': 0,
                'visualizacoes': 1000
            }
        },
        {
            'nome': '⚖️ Condições Normais',
            'estado': {
                'preco_atual': 200,
                'custo_unitario': 80,
                'estoque': 300,
                'demanda_24h': 50,
                'demanda_semanal': np.array([48, 52, 50, 49, 51, 50, 50]),
                'preco_concorrente': 200,
                'precos_outros': np.array([195, 205]),
                'taxa_conversao': 0.05,
                'hora': 16,
                'dia_semana': 3,
                'dia_mes': 20,
                'temporada': 0,  # Normal
                'promocao_ativa': 0,
                'visualizacoes': 3000
            }
        }
    ]

    for cenario in cenarios:
        print(f"\n{cenario['nome']}")
        print("-" * 70)

        decisao = modelo.decide(cenario['estado'], deterministic=True)

        preco = decisao.action['preco']
        desconto = decisao.action['desconto']
        preco_final = preco * (1 - desconto)

        print(f"Preço base: R$ {preco:.2f}")
        print(f"Desconto: {desconto*100:.1f}%")
        print(f"Preço final: R$ {preco_final:.2f}")
        print(f"Promoção relâmpago: {'Sim' if decisao.action['promocao_relampago'] == 1 else 'Não'}")

        destaque_map = {0: 'Nenhum', 1: 'Normal', 2: 'Premium'}
        print(f"Destaque: {destaque_map[decisao.action['destaque']]}")
        print(f"Frete grátis: {'Sim' if decisao.action['frete_gratis'] == 1 else 'Não'}")
        print(f"\nConfiança: {decisao.confidence:.2%}")


if __name__ == "__main__":
    import os
    os.makedirs('./modelos', exist_ok=True)

    print("🚀 Iniciando exemplo de Precificação Dinâmica\n")

    # 1. Treina
    modelo = treinar_modelo_pricing()

    # 2. Testa
    testar_modelo_pricing()

    print("\n" + "=" * 70)
    print("✅ Exemplo concluído!")
    print("=" * 70)
