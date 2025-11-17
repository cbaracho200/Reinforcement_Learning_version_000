"""
🏘️ Exemplo Intermediário 3: Gestão de Portfolio de Imóveis

PROBLEMA REAL:
Você tem uma carteira de 10-20 imóveis e precisa decidir para CADA UM:
- Vender agora ou segurar?
- Alugar ou deixar vazio?
- Reformar ou não?

Considerando:
- Fluxo de caixa mensal
- Valorização esperada
- Custos de manutenção
- Oportunidades de mercado

DECISÃO: O que fazer com cada imóvel do portfolio?

USO:
python examples/intermediate/03_portfolio_imoveis.py
"""

import business_rl as brl
import numpy as np


@brl.problem(name="GestaoPortfolio")
class GestaoPortfolio:
    """
    Problema: Gestão ativa de portfolio de imóveis

    Para CADA imóvel, decide a melhor estratégia no mês
    """

    obs = brl.Dict(
        # Características do imóvel
        valor_mercado=brl.Box(200000, 5000000),
        valor_compra_original=brl.Box(150000, 4000000),
        area_m2=brl.Box(30, 500),
        idade_anos=brl.Box(0, 50),

        # Status atual
        status_atual=brl.Discrete(3, labels=["Vazio", "Alugado", "Venda"]),
        meses_no_status=brl.Box(0, 60),

        # Financeiro
        aluguel_atual_mensal=brl.Box(0, 20000),
        aluguel_mercado_mensal=brl.Box(800, 20000),  # Potencial
        iptu_condominio_mensal=brl.Box(200, 5000),
        custo_manutencao_mensal=brl.Box(100, 2000),

        # Mercado local
        taxa_valorizacao_anual=brl.Box(-0.10, 0.20),  # -10% a +20% ao ano
        tempo_medio_venda_meses=brl.Box(1, 12),
        tempo_medio_locacao_meses=brl.Box(1, 6),
        demanda_aluguel=brl.Box(0, 1),  # 0=baixa, 1=alta
        demanda_compra=brl.Box(0, 1),

        # Condições do imóvel
        estado_conservacao=brl.Discrete(5, labels=[
            "Pessimo", "Ruim", "Regular", "Bom", "Otimo"
        ]),
        precisa_reforma=brl.Discrete(2),
        custo_reforma_estimado=brl.Box(0, 150000),

        # Portfolio
        n_imoveis_vagos=brl.Box(0, 20),
        n_imoveis_alugados=brl.Box(0, 20),
        fluxo_caixa_mensal_total=brl.Box(-50000, 100000),
        liquidez_disponivel=brl.Box(0, 1000000)
    )

    # Decisão para o imóvel
    action = brl.Discrete(5, labels=[
        "Manter",         # Não fazer nada
        "Alugar",         # Colocar para alugar
        "Vender",         # Colocar à venda
        "Reformar",       # Reformar e depois decidir
        "Liquidar"        # Venda rápida (desconto)
    ])

    objectives = brl.Terms(
        fluxo_caixa=0.40,        # Maximizar receita mensal
        valorizacao=0.35,        # Maximizar ganho de capital
        liquidez=0.15,           # Manter liquidez
        diversificacao=0.10      # Balancear portfolio
    )

    def reward_fluxo_caixa(self, state, action, next_state):
        """Maximiza receita mensal."""
        # Receita atual
        aluguel = state['aluguel_atual_mensal']
        custos = state['iptu_condominio_mensal'] + state['custo_manutencao_mensal']
        fluxo_atual = aluguel - custos

        # Se decidir alugar
        if action == 1:  # Alugar
            aluguel_potencial = state['aluguel_mercado_mensal']
            tempo_locacao = state['tempo_medio_locacao_meses']

            # Fluxo esperado (considera vacância)
            fluxo_esperado = (aluguel_potencial - custos) * (12 - tempo_locacao) / 12

            return fluxo_esperado / 100

        # Se decidir vender
        elif action == 2 or action == 4:  # Vender ou Liquidar
            # Perde fluxo de caixa
            return -fluxo_atual / 100

        # Se reformar
        elif action == 3:  # Reformar
            # Custo imediato, mas aumenta aluguel futuro
            custo_reforma = state['custo_reforma_estimado']
            aumento_aluguel = state['aluguel_mercado_mensal'] * 0.20  # +20% após reforma

            # ROI da reforma
            if custo_reforma > 0:
                meses_payback = custo_reforma / aumento_aluguel
                if meses_payback < 24:  # Payback < 2 anos
                    return 50
                else:
                    return -30
            return 0

        # Manter
        else:
            return fluxo_atual / 100

    def reward_valorizacao(self, state, action, next_state):
        """Maximiza ganho de capital."""
        valor_atual = state['valor_mercado']
        valor_compra = state['valor_compra_original']
        taxa_valorizacao = state['taxa_valorizacao_anual']

        # Ganho de capital atual
        ganho_capital = valor_atual - valor_compra
        ganho_percentual = (ganho_capital / valor_compra) * 100

        # Se vender agora
        if action == 2:  # Vender
            # Realiza o ganho
            if ganho_percentual > 20:  # Bom lucro
                return ganho_percentual
            elif ganho_percentual > 10:
                return ganho_percentual * 0.7
            else:
                return -10  # Vendendo cedo demais

        # Se liquidar (venda rápida com desconto)
        elif action == 4:  # Liquidar
            ganho_com_desconto = ganho_percentual * 0.85  # 15% desconto
            if ganho_com_desconto > 0:
                return ganho_com_desconto * 0.5
            else:
                return -30  # Venda forçada com prejuízo

        # Se segurar
        else:
            # Valorizando bem? Segura
            if taxa_valorizacao > 0.10:  # >10% ao ano
                return 30
            elif taxa_valorizacao > 0:
                return 10
            else:
                return -10  # Desvalorizando

    def reward_liquidez(self, state, action, next_state):
        """Mantém liquidez saudável."""
        liquidez = state['liquidez_disponivel']
        fluxo_caixa_total = state['fluxo_caixa_mensal_total']

        # Se vender (gera liquidez)
        if action == 2 or action == 4:  # Vender ou Liquidar
            # Liquidez baixa? Vender é bom
            if liquidez < 100000:
                return 50
            elif liquidez < 300000:
                return 20
            else:
                return -10  # Já tem liquidez suficiente

        # Se reformar (consome liquidez)
        elif action == 3:  # Reformar
            custo_reforma = state['custo_reforma_estimado']
            if custo_reforma > liquidez * 0.5:  # Reforma > 50% liquidez
                return -50  # Muito arriscado
            elif custo_reforma > liquidez * 0.2:
                return -20
            else:
                return 10  # Reforma viável

        # Fluxo de caixa negativo? Precisa liquidez
        if fluxo_caixa_total < 0:
            if action == 2 or action == 4:  # Vender
                return 30
            else:
                return -20

        return 0

    def reward_diversificacao(self, state, action, next_state):
        """Balanceia o portfolio."""
        n_vagos = state['n_imoveis_vagos']
        n_alugados = state['n_imoveis_alugados']
        total_imoveis = n_vagos + n_alugados

        if total_imoveis == 0:
            return 0

        # Proporção de imóveis alugados
        taxa_ocupacao = n_alugados / total_imoveis

        # Ideal: 70-90% ocupado
        if action == 1:  # Alugar
            if taxa_ocupacao < 0.7:
                return 20  # Bom! Aumenta ocupação
            elif taxa_ocupacao < 0.9:
                return 10
            else:
                return 0  # Já está bem ocupado

        # Muita concentração em aluguel? Vender alguns
        if action == 2:  # Vender
            if taxa_ocupacao > 0.9:
                return 15  # Bom! Reduz concentração
            else:
                return 0

        return 0


def treinar_modelo():
    """Treina o modelo de gestão de portfolio."""
    print("="*70)
    print("TREINAMENTO: Gestão de Portfolio de Imóveis")
    print("="*70)

    problema = GestaoPortfolio()

    print("\n📊 Informações do problema:")
    print(f"  Observações: {problema.get_info()['observation_dim']} variáveis")
    print(f"  Decisões: {problema.get_info()['action_space']['action']['n']} ações possíveis")
    print(f"  Objetivos: {problema.get_info()['n_objectives']}")

    print("\n🏋️ Treinando modelo (1 hora)...")
    modelo = brl.train(
        problema,
        algorithm='PPO',
        hours=1,
        config={'learning_rate': 3e-4}
    )

    modelo.save('./modelos/portfolio_imoveis.pt')
    print("\n✅ Modelo salvo!")

    return modelo


def simular_portfolio():
    """Simula gestão de portfolio por 12 meses."""
    print("\n" + "="*70)
    print("SIMULAÇÃO: Gestão de Portfolio por 12 Meses")
    print("="*70)

    modelo = brl.load('./modelos/portfolio_imoveis.pt')

    # Portfolio inicial: 5 imóveis diversos
    portfolio = [
        {
            'nome': 'Apto Centro #1',
            'estado': {
                'valor_mercado': 350000,
                'valor_compra_original': 280000,
                'area_m2': 50,
                'idade_anos': 8,
                'status_atual': 1,  # Alugado
                'meses_no_status': 18,
                'aluguel_atual_mensal': 1800,
                'aluguel_mercado_mensal': 2000,
                'iptu_condominio_mensal': 600,
                'custo_manutencao_mensal': 150,
                'taxa_valorizacao_anual': 0.08,
                'tempo_medio_venda_meses': 4,
                'tempo_medio_locacao_meses': 2,
                'demanda_aluguel': 0.8,
                'demanda_compra': 0.6,
                'estado_conservacao': 3,  # Bom
                'precisa_reforma': 0,
                'custo_reforma_estimado': 0,
                'n_imoveis_vagos': 1,
                'n_imoveis_alugados': 4,
                'fluxo_caixa_mensal_total': 4500,
                'liquidez_disponivel': 150000
            }
        },
        {
            'nome': 'Casa Subúrbio #2',
            'estado': {
                'valor_mercado': 600000,
                'valor_compra_original': 550000,
                'area_m2': 150,
                'idade_anos': 20,
                'status_atual': 0,  # Vazio
                'meses_no_status': 3,
                'aluguel_atual_mensal': 0,
                'aluguel_mercado_mensal': 2500,
                'iptu_condominio_mensal': 400,
                'custo_manutencao_mensal': 300,
                'taxa_valorizacao_anual': 0.05,
                'tempo_medio_venda_meses': 6,
                'tempo_medio_locacao_meses': 3,
                'demanda_aluguel': 0.6,
                'demanda_compra': 0.7,
                'estado_conservacao': 2,  # Regular
                'precisa_reforma': 1,
                'custo_reforma_estimado': 40000,
                'n_imoveis_vagos': 1,
                'n_imoveis_alugados': 4,
                'fluxo_caixa_mensal_total': 4500,
                'liquidez_disponivel': 150000
            }
        },
        {
            'nome': 'Cobertura Luxo #3',
            'estado': {
                'valor_mercado': 1800000,
                'valor_compra_original': 1500000,
                'area_m2': 180,
                'idade_anos': 3,
                'status_atual': 1,  # Alugado
                'meses_no_status': 24,
                'aluguel_atual_mensal': 6500,
                'aluguel_mercado_mensal': 7000,
                'iptu_condominio_mensal': 2000,
                'custo_manutencao_mensal': 500,
                'taxa_valorizacao_anual': 0.12,
                'tempo_medio_venda_meses': 8,
                'tempo_medio_locacao_meses': 2,
                'demanda_aluguel': 0.9,
                'demanda_compra': 0.5,
                'estado_conservacao': 4,  # Ótimo
                'precisa_reforma': 0,
                'custo_reforma_estimado': 0,
                'n_imoveis_vagos': 1,
                'n_imoveis_alugados': 4,
                'fluxo_caixa_mensal_total': 4500,
                'liquidez_disponivel': 150000
            }
        }
    ]

    acoes_labels = ["Manter", "Alugar", "Vender", "Reformar", "Liquidar"]

    print("\n📋 Portfolio inicial: 3 imóveis\n")

    for mes in range(6):  # Simula 6 meses
        print(f"{'='*70}")
        print(f"📅 MÊS {mes + 1}")
        print(f"{'='*70}\n")

        for imovel in portfolio:
            decisao = modelo.decide(imovel['estado'], deterministic=True)
            acao = decisao.action

            print(f"🏠 {imovel['nome']}")
            print(f"   Valor: R$ {imovel['estado']['valor_mercado']:,.0f}")
            print(f"   Status: {['Vazio', 'Alugado', 'À Venda'][imovel['estado']['status_atual']]}")
            if imovel['estado']['aluguel_atual_mensal'] > 0:
                print(f"   Aluguel: R$ {imovel['estado']['aluguel_atual_mensal']:,.0f}/mês")
            print(f"   ✅ DECISÃO: {acoes_labels[acao]}")
            print()

        print()


if __name__ == "__main__":
    import os
    os.makedirs('./modelos', exist_ok=True)

    print("\n🏘️ Gestão Inteligente de Portfolio de Imóveis\n")

    # 1. Treina
    treinar_modelo()

    # 2. Simula
    simular_portfolio()

    print("="*70)
    print("✅ PRONTO!")
    print("   Modelo otimiza portfolio considerando:")
    print("   - Fluxo de caixa mensal")
    print("   - Valorização de longo prazo")
    print("   - Liquidez e risco")
    print("="*70)
