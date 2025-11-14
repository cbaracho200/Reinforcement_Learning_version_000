"""
Exemplo Avançado 3: Gestão de Estoque Multi-Produto

Este exemplo demonstra:
- Múltiplos produtos simultâneos
- Restrições de orçamento e espaço
- Sazonalidade e tendências
- Lead time de fornecedores
- Minimização de rupturas e excessos
"""

import business_rl as brl
import numpy as np


@brl.problem(name="GestaoEstoque")
class GestaoEstoque:
    """
    Problema: Gerenciar estoque de 3 produtos diferentes

    Objetivo: Minimizar custos mantendo disponibilidade
    """

    # ===== OBSERVAÇÕES =====
    obs = brl.Dict(
        # Estoque atual de cada produto
        estoque=brl.Box(0, 500, shape=(3,)),

        # Demanda prevista (próximos 7 dias)
        demanda_prevista=brl.Box(0, 100, shape=(3, 7)),

        # Vendas nos últimos 30 dias
        historico_vendas=brl.Box(0, 100, shape=(3, 30)),

        # Preço de compra de cada produto
        preco_compra=brl.Box(10, 100, shape=(3,)),

        # Preço de venda de cada produto
        preco_venda=brl.Box(20, 200, shape=(3,)),

        # Lead time (dias para entrega)
        lead_time=brl.Discrete(15, offset=1),  # 1-15 dias

        # Pedidos pendentes de entrega
        pedidos_pendentes=brl.Box(0, 500, shape=(3,)),

        # Capacidade de armazenamento disponível
        capacidade_disponivel=brl.Box(0, 1000),

        # Orçamento disponível
        orcamento=brl.Box(0, 50000),

        # Dia do mês
        dia=brl.Discrete(30, offset=1),

        # Mês do ano
        mes=brl.Discrete(12, offset=1),

        # Taxa de obsolescência de cada produto (% por mês)
        taxa_obsolescencia=brl.Box(0, 0.1, shape=(3,))
    )

    # ===== AÇÕES =====
    action = brl.Dict(
        # Quantidade a pedir de cada produto
        quantidade_pedido=brl.Box(0, 200, shape=(3,)),

        # Urgência do pedido (normal=0, expressa=1)
        # Expressa reduz lead time mas custa 30% mais
        urgencia=brl.Discrete(2, labels=["normal", "expressa"]),

        # Fazer promoção para esvaziar estoque?
        promocao=brl.Discrete(2, labels=["nao", "sim"])
    )

    # ===== OBJETIVOS =====
    objectives = brl.Terms(
        custos=0.40,             # 40% minimizar custos
        disponibilidade=0.30,    # 30% manter produtos disponíveis
        otimizacao_capital=0.20, # 20% otimizar capital investido
        obsolescencia=0.10       # 10% evitar produtos obsoletos
    )

    # ===== RESTRIÇÕES =====
    constraints = {
        # Orçamento máximo
        'orcamento': brl.Limit(
            func=lambda s, a: np.sum(a['quantidade_pedido'] * s['preco_compra']),
            max_val=lambda s: s['orcamento'],
            hard=True
        ),

        # Capacidade de armazenamento
        'capacidade': brl.Limit(
            func=lambda s, a: np.sum(s['estoque']) + np.sum(a['quantidade_pedido']),
            max_val=lambda s: s['capacidade_disponivel'] + 1000,
            hard=True
        ),

        # Pedido mínimo por produto (economias de escala)
        'pedido_minimo': brl.Limit(
            func=lambda s, a: np.min(a['quantidade_pedido'][a['quantidade_pedido'] > 0]),
            min_val=10,
            hard=False
        )
    }

    # ===== FUNÇÕES DE RECOMPENSA =====

    def reward_custos(self, state, action, next_state):
        """Minimiza custos de pedido e armazenamento."""
        # Custo de aquisição
        custo_aquisicao = np.sum(
            action['quantidade_pedido'] * state['preco_compra']
        )

        # Custo adicional por urgência
        if action['urgencia'] == 1:  # Expressa
            custo_aquisicao *= 1.30

        # Custo de armazenamento (R$2 por unidade/mês)
        estoque_total = np.sum(state['estoque'])
        custo_armazenamento = estoque_total * 2

        # Custo de promoção
        custo_promocao = 500 if action['promocao'] == 1 else 0

        custo_total = custo_aquisicao + custo_armazenamento + custo_promocao

        return -custo_total / 100  # Negativo para minimizar

    def reward_disponibilidade(self, state, action, next_state):
        """Recompensa por manter produtos disponíveis."""
        # Verifica se haverá ruptura nos próximos 7 dias
        demanda_7dias = np.sum(state['demanda_prevista'], axis=1)
        estoque_projetado = state['estoque'] + action['quantidade_pedido']

        # Penaliza rupturas
        penalidade_ruptura = 0
        for i in range(3):
            if estoque_projetado[i] < demanda_7dias[i]:
                # Ruptura prevista
                falta = demanda_7dias[i] - estoque_projetado[i]
                penalidade_ruptura -= falta * state['preco_venda'][i] * 0.5

        return penalidade_ruptura / 100

    def reward_otimizacao_capital(self, state, action, next_state):
        """Penaliza capital parado em estoque."""
        valor_estoque = np.sum(
            state['estoque'] * state['preco_compra']
        )

        # Penaliza proporcionalmente ao capital imobilizado
        if valor_estoque > 30000:
            penalidade = -(valor_estoque - 30000) / 100
        else:
            penalidade = 0

        return penalidade

    def reward_obsolescencia(self, state, action, next_state):
        """Penaliza produtos que podem ficar obsoletos."""
        # Calcula produtos de baixo giro
        vendas_medias = np.mean(state['historico_vendas'], axis=1)
        dias_estoque = state['estoque'] / (vendas_medias + 1e-6)

        penalidade = 0
        for i in range(3):
            if dias_estoque[i] > 60:
                # Produto parado há muito tempo
                valor_risco = (state['estoque'][i] *
                              state['preco_compra'][i] *
                              state['taxa_obsolescencia'][i])
                penalidade -= valor_risco

        return penalidade / 100


def treinar_modelo_estoque():
    """Treina o modelo de gestão de estoque."""
    print("=" * 70)
    print("TREINAMENTO: GESTÃO DE ESTOQUE")
    print("=" * 70)

    problema = GestaoEstoque()

    print("\n📊 Informações do problema:")
    print(problema.get_info())

    print("\n🏋️ Iniciando treino (2 horas)...")
    modelo = brl.train(
        problema,
        algorithm='PPO',
        hours=2,
        config={
            'learning_rate': 3e-4,
            'batch_size': 256,
            'n_epochs': 10,
            'gamma': 0.995,  # Horizonte mais longo
            'gae_lambda': 0.95
        }
    )

    modelo.save('./modelos/inventory_management.pt')
    print("\n✅ Modelo salvo!")

    return modelo


def testar_modelo_estoque():
    """Testa o modelo com diferentes cenários."""
    print("\n" + "=" * 70)
    print("TESTE: CENÁRIOS DE ESTOQUE")
    print("=" * 70)

    modelo = brl.load('./modelos/inventory_management.pt')

    produtos = ['Produto A', 'Produto B', 'Produto C']

    cenarios = [
        {
            'nome': '🚨 Risco de Ruptura',
            'estado': {
                'estoque': np.array([15, 8, 20]),
                'demanda_prevista': np.array([
                    [10, 12, 11, 10, 15, 13, 12],
                    [8, 9, 10, 8, 9, 10, 8],
                    [5, 5, 6, 5, 5, 6, 5]
                ]),
                'historico_vendas': np.random.randint(5, 15, size=(3, 30)),
                'preco_compra': np.array([30, 45, 25]),
                'preco_venda': np.array([60, 90, 50]),
                'lead_time': 5,
                'pedidos_pendentes': np.array([0, 0, 0]),
                'capacidade_disponivel': 800,
                'orcamento': 10000,
                'dia': 15,
                'mes': 6,
                'taxa_obsolescencia': np.array([0.02, 0.01, 0.05])
            }
        },
        {
            'nome': '📦 Excesso de Estoque',
            'estado': {
                'estoque': np.array([450, 380, 420]),
                'demanda_prevista': np.array([
                    [8, 7, 8, 9, 8, 7, 8],
                    [5, 6, 5, 5, 6, 5, 6],
                    [3, 3, 4, 3, 3, 3, 4]
                ]),
                'historico_vendas': np.random.randint(3, 10, size=(3, 30)),
                'preco_compra': np.array([30, 45, 25]),
                'preco_venda': np.array([60, 90, 50]),
                'lead_time': 7,
                'pedidos_pendentes': np.array([100, 50, 80]),
                'capacidade_disponivel': 200,
                'orcamento': 5000,
                'dia': 5,
                'mes': 3,
                'taxa_obsolescencia': np.array([0.02, 0.01, 0.05])
            }
        }
    ]

    for cenario in cenarios:
        print(f"\n{cenario['nome']}")
        print("-" * 70)

        estado = cenario['estado']
        decisao = modelo.decide(estado, deterministic=True)

        print("\n📊 Situação Atual:")
        for i, produto in enumerate(produtos):
            print(f"  {produto}: {estado['estoque'][i]} unidades")

        print(f"\nOrçamento: R$ {estado['orcamento']:,.2f}")
        print(f"Capacidade livre: {estado['capacidade_disponivel']} unidades")

        print("\n📝 Decisão do Modelo:")
        for i, produto in enumerate(produtos):
            qtd = decisao.action['quantidade_pedido'][i]
            if qtd > 0:
                custo = qtd * estado['preco_compra'][i]
                print(f"  {produto}: Pedir {qtd:.0f} unidades (R$ {custo:,.2f})")

        urgencia = "Expressa" if decisao.action['urgencia'] == 1 else "Normal"
        promocao = "Sim" if decisao.action['promocao'] == 1 else "Não"

        print(f"\nUrgência: {urgencia}")
        print(f"Fazer promoção: {promocao}")
        print(f"Confiança: {decisao.confidence:.2%}")


if __name__ == "__main__":
    import os
    os.makedirs('./modelos', exist_ok=True)

    print("🚀 Iniciando exemplo de Gestão de Estoque\n")

    # 1. Treina
    modelo = treinar_modelo_estoque()

    # 2. Testa
    testar_modelo_estoque()

    print("\n" + "=" * 70)
    print("✅ Exemplo concluído!")
    print("=" * 70)
