"""
🎯 Exemplo Intermediário 5: Estratégia de Vendas - Lançamento Imobiliário

PROBLEMA REAL:
Você lançou um empreendimento (prédio com 50 apartamentos) e precisa
decidir SEMANALMENTE:
- Preço dos apartamentos
- Desconto a oferecer
- Investimento em marketing
- Tabelas de pagamento

Para:
- Vender rápido (gerar caixa)
- Maximizar receita total
- Manter a imagem do empreendimento

DECISÃO: Estratégia comercial da semana

USO:
python examples/intermediate/05_vendas_lancamento.py
"""

import business_rl as brl
import numpy as np


@brl.problem(name="VendasLancamento")
class VendasLancamento:
    """
    Problema: Estratégia de vendas de empreendimento imobiliário

    A cada semana, decide pricing, descontos e investimento em marketing
    """

    obs = brl.Dict(
        # Status do empreendimento
        unidades_totais=brl.Box(20, 200),
        unidades_vendidas=brl.Box(0, 200),
        unidades_reservadas=brl.Box(0, 50),
        percentual_vendido=brl.Box(0, 1),  # 0% a 100%

        # Tempo
        semanas_desde_lancamento=brl.Box(0, 104),  # Até 2 anos
        percentual_obra_concluido=brl.Box(0, 1),

        # Vendas recentes
        vendas_ultima_semana=brl.Box(0, 20),
        vendas_ultimas_4_semanas=brl.Box(0, 50),
        velocidade_vendas=brl.Box(0, 1),  # unidades/semana normalizado

        # Pricing
        preco_medio_atual=brl.Box(200000, 2000000),
        preco_lancamento_original=brl.Box(200000, 2000000),
        desconto_atual_percentual=brl.Box(0, 0.30),  # 0% a 30%

        # Concorrência
        lancamentos_concorrentes_regiao=brl.Box(0, 10),
        preco_medio_concorrencia=brl.Box(200000, 2000000),
        nossa_competitividade=brl.Box(0.5, 1.5),  # preço nosso / preço concorrente

        # Demanda
        visitas_stand_semana=brl.Box(0, 200),
        taxa_conversao=brl.Box(0, 0.50),  # visitas → vendas
        interesse_mercado=brl.Box(0, 1),  # indicador de demanda

        # Marketing
        investimento_marketing_mes=brl.Box(0, 500000),
        leads_gerados_semana=brl.Box(0, 300),
        custo_por_lead=brl.Box(50, 500),

        # Financeiro
        vgv_total=brl.Box(10000000, 500000000),  # Valor Geral de Vendas
        receita_acumulada=brl.Box(0, 500000000),
        meta_receita_mensal=brl.Box(1000000, 50000000),
        fluxo_caixa_disponivel=brl.Box(0, 100000000),

        # Produto
        qualidade_produto=brl.Discrete(5, labels=[
            "Economico", "Medio", "Medio_Alto", "Alto", "Luxo"
        ]),
        diferenciais=brl.Box(0, 10),  # Número de diferenciais

        # Sazonalidade
        mes_ano=brl.Discrete(12),
        periodo_favoravel_vendas=brl.Discrete(2)  # Alta temporada?
    )

    # Estratégia comercial
    action = brl.Dict(
        # Ajuste de preço
        ajuste_preco=brl.Box(-0.10, 0.10),  # -10% a +10%

        # Desconto promocional
        desconto_promocional=brl.Box(0, 0.25),  # 0% a 25%

        # Investimento em marketing (% do VGV mensal)
        investimento_marketing=brl.Box(0, 0.05),  # 0% a 5%

        # Condições de pagamento
        flexibilidade_pagamento=brl.Discrete(3, labels=[
            "Rigorosa",    # À vista ou pouca flexibilidade
            "Moderada",    # Condições normais de mercado
            "Flexivel"     # Máxima flexibilidade
        ])
    )

    objectives = brl.Terms(
        receita_total=0.40,       # Maximizar receita
        velocidade_vendas=0.30,   # Vender rápido
        rentabilidade=0.20,       # Margem por unidade
        posicionamento=0.10       # Manter imagem
    )

    def reward_receita_total(self, state, action, next_state):
        """Maximiza receita total."""
        preco_atual = state['preco_medio_atual']
        ajuste = action['ajuste_preco']
        desconto = action['desconto_promocional']

        # Novo preço
        preco_novo = preco_atual * (1 + ajuste) * (1 - desconto)

        # Unidades que pode vender
        unidades_disponiveis = state['unidades_totais'] - state['unidades_vendidas']

        # Estimativa de vendas baseada no preço
        competitividade_base = state['nossa_competitividade']
        nossa_nova_competitividade = preco_novo / state['preco_medio_concorrencia']

        # Quanto mais competitivo, mais vende
        fator_demanda = max(0, 2.0 - nossa_nova_competitividade) * state['interesse_mercado']

        vendas_estimadas = min(
            state['visitas_stand_semana'] * state['taxa_conversao'] * fator_demanda,
            unidades_disponiveis
        )

        receita_estimada = vendas_estimadas * preco_novo

        return receita_estimada / 100000  # Normaliza

    def reward_velocidade_vendas(self, state, action, next_state):
        """Recompensa vender rápido."""
        percentual_vendido = state['percentual_vendido']
        semanas = state['semanas_desde_lancamento']
        desconto = action['desconto_promocional']
        flex_pagamento = action['flexibilidade_pagamento']

        # Meta: vender 80% em 1 ano (52 semanas)
        percentual_esperado = min(0.8, semanas / 52 * 0.8)
        diferenca = percentual_vendido - percentual_esperado

        # Atrasado nas vendas?
        if diferenca < -0.15:  # >15% atrasado
            # Precisa agressividade
            if desconto > 0.15 or flex_pagamento == 2:
                return 60  # Bom! Está tentando recuperar
            else:
                return -40  # Deveria ser mais agressivo

        # Vendendo muito rápido?
        elif diferenca > 0.15:  # >15% adiantado
            # Pode segurar preço
            if desconto < 0.05:
                return 40  # Ótimo! Vendendo bem sem desconto
            else:
                return -20  # Dando desconto desnecessário

        # No ritmo certo
        else:
            if 0.05 <= desconto <= 0.10:
                return 30  # Desconto moderado
            else:
                return 10

    def reward_rentabilidade(self, state, action, next_state):
        """Mantém margem de lucro."""
        preco_lancamento = state['preco_lancamento_original']
        preco_atual = state['preco_medio_atual']
        ajuste = action['ajuste_preco']
        desconto = action['desconto_promocional']
        invest_marketing = action['investimento_marketing']

        # Preço final
        preco_final = preco_atual * (1 + ajuste) * (1 - desconto)

        # Margem vs preço original
        margem = (preco_final / preco_lancamento - 1) * 100

        # Custo de marketing
        custo_marketing_percentual = invest_marketing

        # Margem líquida
        margem_liquida = margem - (custo_marketing_percentual * 100)

        # Penaliza vender abaixo do lançamento
        if margem < -10:  # >10% abaixo
            return -80
        elif margem < -5:
            return -40
        elif margem < 0:
            return -10
        elif margem < 5:
            return 10
        elif margem < 10:
            return 30
        else:
            return 50  # Ótima margem!

    def reward_posicionamento(self, state, action, next_state):
        """Mantém posicionamento de mercado."""
        qualidade = state['qualidade_produto']
        desconto = action['desconto_promocional']
        nossa_competitividade = state['nossa_competitividade']
        ajuste = action['ajuste_preco']

        # Produto de luxo não deve dar muito desconto
        if qualidade >= 3:  # Alto padrão ou Luxo
            if desconto > 0.15:
                return -50  # Desconto excessivo prejudica imagem
            elif desconto > 0.10:
                return -20
            elif desconto < 0.05:
                return 30  # Mantém exclusividade

        # Produto econômico: competitividade é crucial
        elif qualidade <= 1:
            if nossa_competitividade > 1.1:  # 10% mais caro
                return -30  # Precisa ser mais competitivo
            elif nossa_competitividade < 0.95:
                return 20  # Boa competitividade

        # Aumentar preço sem justificativa
        if ajuste > 0.05 and state['velocidade_vendas'] < 0.5:
            return -40  # Vendendo devagar e aumentando preço?

        return 0


def treinar_modelo():
    """Treina o modelo de vendas."""
    print("="*70)
    print("TREINAMENTO: Estratégia de Vendas de Lançamento")
    print("="*70)

    problema = VendasLancamento()

    print("\n📊 Informações do problema:")
    print(f"  Observações: {problema.get_info()['observation_dim']} variáveis")
    print(f"  Ações: Híbridas (preço + desconto + marketing + condições)")
    print(f"  Objetivos: {problema.get_info()['n_objectives']}")

    print("\n🏋️ Treinando modelo (1.5 horas)...")
    modelo = brl.train(
        problema,
        algorithm='PPO',  # PPO funciona bem com ações híbridas
        hours=1.5,
        config={'learning_rate': 3e-4}
    )

    modelo.save('./modelos/vendas_lancamento.pt')
    print("\n✅ Modelo salvo!")

    return modelo


def simular_vendas():
    """Simula campanha de vendas."""
    print("\n" + "="*70)
    print("SIMULAÇÃO: Campanha de Vendas - 12 Semanas")
    print("="*70)

    modelo = brl.load('./modelos/vendas_lancamento.pt')

    # Empreendimento
    print("\n🏢 Empreendimento: Residencial Alto Padrão")
    print("   80 apartamentos | VGV: R$ 64M | Ticket médio: R$ 800k\n")

    # Simula 4 momentos diferentes
    momentos = [
        {
            'semana': 1,
            'nome': 'Lançamento',
            'estado': {
                'unidades_totais': 80,
                'unidades_vendidas': 0,
                'unidades_reservadas': 5,
                'percentual_vendido': 0,
                'semanas_desde_lancamento': 1,
                'percentual_obra_concluido': 0.10,
                'vendas_ultima_semana': 0,
                'vendas_ultimas_4_semanas': 0,
                'velocidade_vendas': 0,
                'preco_medio_atual': 800000,
                'preco_lancamento_original': 800000,
                'desconto_atual_percentual': 0,
                'lancamentos_concorrentes_regiao': 3,
                'preco_medio_concorrencia': 850000,
                'nossa_competitividade': 0.94,
                'visitas_stand_semana': 80,
                'taxa_conversao': 0.15,
                'interesse_mercado': 0.8,
                'investimento_marketing_mes': 200000,
                'leads_gerados_semana': 150,
                'custo_por_lead': 200,
                'vgv_total': 64000000,
                'receita_acumulada': 0,
                'meta_receita_mensal': 8000000,
                'fluxo_caixa_disponivel': 15000000,
                'qualidade_produto': 3,  # Alto padrão
                'diferenciais': 7,
                'mes_ano': 9,
                'periodo_favoravel_vendas': 1
            }
        },
        {
            'semana': 12,
            'nome': '3 Meses - Ritmo Lento',
            'estado': {
                'unidades_totais': 80,
                'unidades_vendidas': 18,
                'unidades_reservadas': 3,
                'percentual_vendido': 0.225,  # 22.5%
                'semanas_desde_lancamento': 12,
                'percentual_obra_concluido': 0.25,
                'vendas_ultima_semana': 1,
                'vendas_ultimas_4_semanas': 4,
                'velocidade_vendas': 0.3,
                'preco_medio_atual': 820000,
                'preco_lancamento_original': 800000,
                'desconto_atual_percentual': 0,
                'lancamentos_concorrentes_regiao': 4,
                'preco_medio_concorrencia': 830000,
                'nossa_competitividade': 0.99,
                'visitas_stand_semana': 45,
                'taxa_conversao': 0.10,
                'interesse_mercado': 0.6,
                'investimento_marketing_mes': 180000,
                'leads_gerados_semana': 90,
                'custo_por_lead': 220,
                'vgv_total': 64000000,
                'receita_acumulada': 14760000,
                'meta_receita_mensal': 8000000,
                'fluxo_caixa_disponivel': 12000000,
                'qualidade_produto': 3,
                'diferenciais': 7,
                'mes_ano': 11,
                'periodo_favoravel_vendas': 0
            }
        },
        {
            'semana': 26,
            'nome': '6 Meses - Reta Final',
            'estado': {
                'unidades_totais': 80,
                'unidades_vendidas': 58,
                'unidades_reservadas': 2,
                'percentual_vendido': 0.725,
                'semanas_desde_lancamento': 26,
                'percentual_obra_concluido': 0.60,
                'vendas_ultima_semana': 2,
                'vendas_ultimas_4_semanas': 8,
                'velocidade_vendas': 0.5,
                'preco_medio_atual': 850000,
                'preco_lancamento_original': 800000,
                'desconto_atual_percentual': 0,
                'lancamentos_concorrentes_regiao': 5,
                'preco_medio_concorrencia': 870000,
                'nossa_competitividade': 0.98,
                'visitas_stand_semana': 30,
                'taxa_conversao': 0.18,
                'interesse_mercado': 0.7,
                'investimento_marketing_mes': 120000,
                'leads_gerados_semana': 60,
                'custo_por_lead': 200,
                'vgv_total': 64000000,
                'receita_acumulada': 49300000,
                'meta_receita_mensal': 4000000,
                'fluxo_caixa_disponivel': 8000000,
                'qualidade_produto': 3,
                'diferenciais': 7,
                'mes_ano': 3,
                'periodo_favoravel_vendas': 1
            }
        }
    ]

    for momento in momentos:
        print(f"{'='*70}")
        print(f"📅 Semana {momento['semana']} - {momento['nome']}")
        print(f"{'='*70}")

        estado = momento['estado']

        # Modelo decide estratégia
        decisao = modelo.decide(estado, deterministic=True)
        acao = decisao.action

        print(f"\n📊 Situação:")
        print(f"  Vendidas: {estado['unidades_vendidas']}/{estado['unidades_totais']} ({estado['percentual_vendido']:.0%})")
        print(f"  Velocidade: {estado['vendas_ultimas_4_semanas']:.0f} unidades/mês")
        print(f"  Visitas: {estado['visitas_stand_semana']:.0f}/semana | Conversão: {estado['taxa_conversao']:.0%}")
        print(f"  Receita acumulada: R$ {estado['receita_acumulada']:,.0f}")

        print(f"\n💰 Pricing:")
        print(f"  Preço atual: R$ {estado['preco_medio_atual']:,.0f}")
        print(f"  vs Lançamento: {(estado['preco_medio_atual']/estado['preco_lancamento_original']-1)*100:+.1f}%")
        print(f"  vs Concorrência: {(estado['nossa_competitividade']-1)*100:+.1f}%")

        print(f"\n✅ ESTRATÉGIA DA SEMANA:")
        print(f"  Ajuste de preço: {acao['ajuste_preco']:+.1%}")
        print(f"  Desconto promocional: {acao['desconto_promocional']:.1%}")
        print(f"  Investimento marketing: {acao['investimento_marketing']:.1%} do VGV")
        print(f"  Pagamento: {['Rigoroso', 'Moderado', 'Flexível'][acao['flexibilidade_pagamento']]}")

        # Preço final
        preco_final = estado['preco_medio_atual'] * (1 + acao['ajuste_preco']) * (1 - acao['desconto_promocional'])
        print(f"\n  💡 Preço efetivo: R$ {preco_final:,.0f}")

        # Justificativa
        if momento['semana'] == 1:
            print(f"     Lançamento agressivo para gerar movimento inicial")
        elif estado['velocidade_vendas'] < 0.4:
            print(f"     Acelerando vendas com condições atrativas")
        else:
            print(f"     Mantendo rentabilidade com vendas consistentes")

        print()


if __name__ == "__main__":
    import os
    os.makedirs('./modelos', exist_ok=True)

    print("\n🎯 Estratégia Inteligente de Vendas - Lançamento Imobiliário\n")

    # 1. Treina
    treinar_modelo()

    # 2. Simula
    simular_vendas()

    print("="*70)
    print("✅ PRONTO!")
    print("   Modelo otimiza vendas considerando:")
    print("   - Receita total e velocidade")
    print("   - Rentabilidade por unidade")
    print("   - Posicionamento de mercado")
    print("   - Condições competitivas")
    print("="*70)
