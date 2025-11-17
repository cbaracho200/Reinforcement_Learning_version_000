"""
🏗️ Exemplo Básico 2: Compra de Materiais de Construção

PROBLEMA REAL:
Você é gestor de obras e precisa decidir QUANDO e QUANTO comprar
de materiais (cimento, areia, ferro, etc.) para:
- Não faltar material (obra não para)
- Não ter estoque excessivo (custo de armazenamento)
- Aproveitar promoções e preços baixos

DECISÃO: Quanto comprar hoje?

USO:
python examples/basic/02_compra_materiais.py
"""

import business_rl as brl
import numpy as np


@brl.problem(name="CompraMateriais")
class CompraMateriais:
    """
    Problema: Decidir quantidade de material a comprar

    Observações: Estoque, consumo, preços, cronograma
    Ação: Quantidade a comprar (em toneladas)
    Objetivos: Minimizar custo e evitar falta
    """

    obs = brl.Dict(
        # Estoque atual
        estoque_atual_ton=brl.Box(0, 100),  # Toneladas em estoque
        espaco_armazem_ton=brl.Box(20, 200),  # Capacidade máxima

        # Consumo
        consumo_medio_dia_ton=brl.Box(0.5, 10),  # Quanto usa por dia
        consumo_semana_passada_ton=brl.Box(3, 70),  # Histórico
        previsao_consumo_7d_ton=brl.Box(3, 70),  # Previsão próximos 7 dias

        # Cronograma da obra
        fase_obra=brl.Discrete(5, labels=[
            "Fundacao",      # 0 - Alto consumo
            "Estrutura",     # 1 - Altíssimo consumo
            "Alvenaria",     # 2 - Médio consumo
            "Acabamento",    # 3 - Baixo consumo
            "Finalizacao"    # 4 - Muito baixo
        ]),
        dias_ate_proxima_fase=brl.Box(0, 60),
        percentual_obra_concluido=brl.Box(0, 1),  # 0% a 100%

        # Preços e fornecedores
        preco_atual_ton=brl.Box(300, 1500),  # R$/tonelada
        preco_medio_30d_ton=brl.Box(300, 1500),
        em_promocao=brl.Discrete(2),  # 0=não, 1=sim
        desconto_volume=brl.Box(0, 0.20),  # 0% a 20% desconto

        # Financeiro
        orcamento_disponivel=brl.Box(0, 500000),
        prazo_pagamento_dias=brl.Discrete(60),  # 0 a 59 dias

        # Logística
        prazo_entrega_dias=brl.Discrete(15),  # 0 a 14 dias
        custo_armazenamento_ton_mes=brl.Box(10, 100)
    )

    # Decide quanto comprar (em toneladas)
    action = brl.Box(0, 50)  # 0 a 50 toneladas

    # Objetivos
    objectives = brl.Terms(
        custo_total=0.50,        # Minimizar custo
        disponibilidade=0.35,    # Garantir material disponível
        aproveitamento=0.15      # Aproveitar promoções
    )

    def reward_custo_total(self, state, action, next_state):
        """Minimiza custo total (compra + armazenamento)."""
        quantidade = action
        preco_ton = state['preco_atual_ton']

        # Aplica desconto por volume
        desconto = state['desconto_volume']
        if quantidade > 20:
            preco_final = preco_ton * (1 - desconto)
        else:
            preco_final = preco_ton

        # Custo de compra
        custo_compra = quantidade * preco_final

        # Custo de armazenamento (estoque médio por mês)
        estoque_medio = (state['estoque_atual_ton'] + quantidade) / 2
        custo_armazem_mes = estoque_medio * state['custo_armazenamento_ton_mes']

        # Custo total mensal
        custo_total = custo_compra + custo_armazem_mes

        # Normaliza e inverte (queremos MINIMIZAR)
        return -custo_total / 1000

    def reward_disponibilidade(self, state, action, next_state):
        """Garante que não vai faltar material."""
        quantidade_comprada = action
        estoque_atual = state['estoque_atual_ton']
        consumo_previsto_7d = state['previsao_consumo_7d_ton']
        prazo_entrega = state['prazo_entrega_dias']

        # Estoque após compra (considerando prazo de entrega)
        estoque_futuro = estoque_atual + quantidade_comprada
        consumo_durante_entrega = state['consumo_medio_dia_ton'] * prazo_entrega

        # Estoque disponível após entrega
        estoque_disponivel = estoque_futuro - consumo_durante_entrega

        # Cobertura em dias
        if state['consumo_medio_dia_ton'] > 0:
            dias_cobertura = estoque_disponivel / state['consumo_medio_dia_ton']
        else:
            dias_cobertura = 999

        # Penaliza se estoque muito baixo ou muito alto
        if dias_cobertura < 3:
            return -100  # Crítico! Vai faltar
        elif dias_cobertura < 7:
            return -30  # Arriscado
        elif dias_cobertura < 15:
            return 50  # Ideal!
        elif dias_cobertura < 30:
            return 20  # OK
        else:
            return -20  # Estoque excessivo

    def reward_aproveitamento(self, state, action, next_state):
        """Aproveita promoções e preços baixos."""
        quantidade = action
        preco_atual = state['preco_atual_ton']
        preco_medio = state['preco_medio_30d_ton']
        em_promocao = state['em_promocao']

        # Preço está bom?
        ratio_preco = preco_atual / preco_medio

        # Está em promoção?
        if em_promocao == 1:
            if quantidade > 10:  # Compra volume em promoção
                return 50
            elif quantidade > 5:
                return 30
            else:
                return 0  # Perdeu oportunidade

        # Preço baixo (não-promoção)
        elif ratio_preco < 0.9:  # 10% abaixo da média
            if quantidade > 10:
                return 30
            else:
                return 10

        # Preço alto
        elif ratio_preco > 1.1:
            if quantidade > 10:
                return -30  # Comprou muito no preço alto
            elif quantidade > 0:
                return -10
            else:
                return 0  # Bom, não comprou

        return 0


def treinar_modelo():
    """Treina o modelo de compra."""
    print("="*70)
    print("TREINAMENTO: Compra Inteligente de Materiais")
    print("="*70)

    problema = CompraMateriais()

    print("\n📊 Informações do problema:")
    print(f"  Observações: {problema.get_info()['observation_dim']} variáveis")
    print(f"  Ação: Quantidade a comprar (0-50 toneladas)")
    print(f"  Objetivos: {problema.get_info()['n_objectives']}")

    print("\n🏋️ Treinando modelo (30 minutos)...")
    modelo = brl.train(
        problema,
        algorithm='SAC',
        hours=0.5,
        config={'learning_rate': 3e-4}
    )

    modelo.save('./modelos/compra_materiais.pt')
    print("\n✅ Modelo salvo!")

    return modelo


def testar_modelo():
    """Testa o modelo com cenários reais."""
    print("\n" + "="*70)
    print("TESTE: Decisões de Compra")
    print("="*70)

    modelo = brl.load('./modelos/compra_materiais.pt')

    # Cenários de teste
    cenarios = [
        {
            'nome': 'Fase Estrutura - Estoque Baixo',
            'situacao': {
                'estoque_atual_ton': 5,
                'espaco_armazem_ton': 100,
                'consumo_medio_dia_ton': 3,
                'consumo_semana_passada_ton': 21,
                'previsao_consumo_7d_ton': 21,
                'fase_obra': 1,  # Estrutura
                'dias_ate_proxima_fase': 30,
                'percentual_obra_concluido': 0.35,
                'preco_atual_ton': 800,
                'preco_medio_30d_ton': 850,
                'em_promocao': 0,
                'desconto_volume': 0.10,
                'orcamento_disponivel': 150000,
                'prazo_pagamento_dias': 30,
                'prazo_entrega_dias': 5,
                'custo_armazenamento_ton_mes': 50
            }
        },
        {
            'nome': 'Promoção! - Estoque Médio',
            'situacao': {
                'estoque_atual_ton': 15,
                'espaco_armazem_ton': 80,
                'consumo_medio_dia_ton': 2,
                'consumo_semana_passada_ton': 14,
                'previsao_consumo_7d_ton': 14,
                'fase_obra': 2,  # Alvenaria
                'dias_ate_proxima_fase': 20,
                'percentual_obra_concluido': 0.60,
                'preco_atual_ton': 650,
                'preco_medio_30d_ton': 800,
                'em_promocao': 1,  # PROMOÇÃO!
                'desconto_volume': 0.15,
                'orcamento_disponivel': 200000,
                'prazo_pagamento_dias': 45,
                'prazo_entrega_dias': 3,
                'custo_armazenamento_ton_mes': 40
            }
        },
        {
            'nome': 'Acabamento - Estoque Alto',
            'situacao': {
                'estoque_atual_ton': 35,
                'espaco_armazem_ton': 60,
                'consumo_medio_dia_ton': 0.8,
                'consumo_semana_passada_ton': 5.6,
                'previsao_consumo_7d_ton': 5.6,
                'fase_obra': 3,  # Acabamento
                'dias_ate_proxima_fase': 45,
                'percentual_obra_concluido': 0.85,
                'preco_atual_ton': 900,
                'preco_medio_30d_ton': 800,
                'em_promocao': 0,
                'desconto_volume': 0.08,
                'orcamento_disponivel': 80000,
                'prazo_pagamento_dias': 15,
                'prazo_entrega_dias': 7,
                'custo_armazenamento_ton_mes': 60
            }
        }
    ]

    print("\n📋 Analisando 3 situações diferentes:\n")

    for cenario in cenarios:
        print(f"{'='*70}")
        print(f"📍 {cenario['nome']}")
        print(f"{'='*70}")

        situacao = cenario['situacao']

        # Modelo decide
        decisao = modelo.decide(situacao, deterministic=True)
        quantidade_comprar = decisao.action

        # Análise
        estoque = situacao['estoque_atual_ton']
        consumo_dia = situacao['consumo_medio_dia_ton']
        dias_cobertura_atual = estoque / consumo_dia if consumo_dia > 0 else 999
        dias_cobertura_pos_compra = (estoque + quantidade_comprar) / consumo_dia if consumo_dia > 0 else 999
        custo_compra = quantidade_comprar * situacao['preco_atual_ton']

        print(f"\n📊 Situação Atual:")
        print(f"  Estoque: {estoque:.1f} ton (cobertura: {dias_cobertura_atual:.0f} dias)")
        print(f"  Consumo médio: {consumo_dia:.1f} ton/dia")
        print(f"  Fase: {['Fundação', 'Estrutura', 'Alvenaria', 'Acabamento', 'Finalização'][situacao['fase_obra']]}")
        print(f"  Obra: {situacao['percentual_obra_concluido']:.0%} concluída")

        print(f"\n💰 Condições de Mercado:")
        print(f"  Preço: R$ {situacao['preco_atual_ton']:.2f}/ton")
        print(f"  vs Média 30d: {(situacao['preco_atual_ton']/situacao['preco_medio_30d_ton']-1)*100:+.1f}%")
        if situacao['em_promocao'] == 1:
            print(f"  🎉 PROMOÇÃO ATIVA!")
        print(f"  Desconto volume: {situacao['desconto_volume']:.0%}")

        print(f"\n✅ DECISÃO: Comprar {quantidade_comprar:.1f} toneladas")
        print(f"  Custo total: R$ {custo_compra:,.2f}")
        print(f"  Novo estoque: {estoque + quantidade_comprar:.1f} ton")
        print(f"  Nova cobertura: {dias_cobertura_pos_compra:.0f} dias")

        # Recomendação
        if quantidade_comprar < 1:
            recomendacao = "Não comprar agora (estoque suficiente ou preço alto)"
        elif quantidade_comprar < 10:
            recomendacao = "Compra moderada"
        elif quantidade_comprar < 25:
            recomendacao = "Compra significativa"
        else:
            recomendacao = "Compra grande (aproveitar promoção/preço baixo)"

        print(f"  📝 {recomendacao}")
        print()


if __name__ == "__main__":
    import os
    os.makedirs('./modelos', exist_ok=True)

    print("\n🏗️ Compra Inteligente de Materiais de Construção\n")

    # 1. Treina
    treinar_modelo()

    # 2. Testa
    testar_modelo()

    print("="*70)
    print("✅ PRONTO!")
    print("   Modelo otimiza compras considerando:")
    print("   - Necessidade da obra")
    print("   - Oportunidades de preço")
    print("   - Custos de armazenamento")
    print("="*70)
