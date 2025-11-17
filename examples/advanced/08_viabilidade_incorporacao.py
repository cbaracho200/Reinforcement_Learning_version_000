"""
💼 Exemplo Avançado 8: Análise de Viabilidade de Incorporação

PROBLEMA REAL:
Você encontrou um terreno e precisa decidir se vale a pena incorporar.
Análise completa considerando:
- TIR (Taxa Interna de Retorno)
- ROI (Return on Investment)
- Payback
- Risco (CVaR - Conditional Value at Risk)

DECISÃO: Comprar, Negociar ou Aguardar?

DIFERENÇA vs exemplos básicos:
- Múltiplos objetivos simultâneos (TIR + ROI + Payback)
- Gestão de risco com CVaR
- Simulação de cenários (otimista, realista, pessimista)

USO:
python examples/advanced/08_viabilidade_incorporacao.py
"""

import business_rl as brl
import numpy as np


@brl.problem(name="ViabilidadeIncorporacao")
class ViabilidadeIncorporacao:
    """
    Problema: Análise de viabilidade de incorporação imobiliária

    Avalia se um terreno/projeto vale a pena desenvolver
    """

    obs = brl.Dict(
        # Terreno
        preco_terreno_m2=brl.Box(500, 5000),
        area_terreno_m2=brl.Box(500, 20000),
        area_construida_permitida_m2=brl.Box(1000, 50000),

        # Projeto
        vgv_m2=brl.Box(4000, 15000),  # Valor Geral de Vendas por m²
        custo_obra_m2=brl.Box(2000, 6000),
        prazo_obra_meses=brl.Box(12, 48),
        prazo_vendas_meses=brl.Box(6, 36),

        # Curva de vendas (% vendido em cada fase)
        vendas_durante_obra=brl.Box(0.30, 0.70),  # 30% a 70% durante obra
        vendas_pos_obra=brl.Box(0.20, 0.60),      # 20% a 60% após entrega

        # Tabela de pagamento
        entrada_percentual=brl.Box(0.10, 0.30),
        mensais_durante_obra_perc=brl.Box(0.20, 0.50),
        saldo_na_chave_perc=brl.Box(0.30, 0.60),

        # Custos e taxas
        taxa_incorporacao=brl.Box(0.05, 0.15),      # 5% a 15% do VGV
        taxa_corretagem=brl.Box(0.04, 0.08),        # 4% a 8% do VGV
        taxa_marketing=brl.Box(0.02, 0.05),         # 2% a 5% do VGV
        impostos_percentual=brl.Box(0.03, 0.08),    # Impostos

        # Mercado
        absorcao_mercado_un_mes=brl.Box(2, 20),     # Unidades/mês que mercado absorve
        preco_m2_concorrencia=brl.Box(4000, 15000),
        imoveis_estoque_regiao=brl.Box(0, 200),

        # Financiamento
        taxa_juros_anual=brl.Box(0.08, 0.18),       # 8% a 18% ao ano
        pode_financiar_terreno=brl.Discrete(2),
        pode_financiar_obra=brl.Discrete(2),

        # Cenário econômico
        cenario=brl.Discrete(3, labels=["Pessimista", "Realista", "Otimista"]),
        inflacao_anual=brl.Box(0.03, 0.10),
        renda_media_regiao=brl.Box(3000, 20000)
    )

    # Decisão
    action = brl.Discrete(3, labels=[
        "Aguardar",    # Não fazer agora
        "Negociar",    # Tentar melhorar condições
        "Comprar"      # Executar o projeto
    ])

    # Múltiplos objetivos com pesos
    objectives = brl.Terms(
        TIR=0.40,          # Taxa Interna de Retorno
        ROI=0.30,          # Return on Investment
        payback=0.20,      # Tempo de retorno
        risco=0.10         # Gestão de risco
    )

    # Gestão de risco com CVaR
    risk = brl.CVaR(
        alpha=0.05,         # Analisa 5% piores cenários
        max_drawdown=0.30   # Perda máxima aceitável: 30%
    )

    def reward_TIR(self, state, action, next_state):
        """Maximiza Taxa Interna de Retorno."""
        # Calcula TIR estimada
        tir = self._calcular_tir(state)

        # Meta: TIR > 18% ao ano
        if action == 2:  # Comprar
            if tir > 0.20:  # TIR > 20%
                return 100
            elif tir > 0.18:
                return 80
            elif tir > 0.15:
                return 50
            elif tir > 0.12:
                return 20
            else:
                return -50  # TIR muito baixa

        # Negociar se TIR marginal (12-15%)
        elif action == 1:  # Negociar
            if 0.12 <= tir <= 0.15:
                return 60  # Boa decisão negociar
            else:
                return 0

        # Aguardar se TIR ruim
        else:  # Aguardar
            if tir < 0.12:
                return 40  # Correto aguardar
            else:
                return -30  # Perdendo oportunidade

    def reward_ROI(self, state, action, next_state):
        """Maximiza ROI (retorno sobre investimento)."""
        roi = self._calcular_roi(state)

        # Meta: ROI > 25%
        if action == 2:  # Comprar
            if roi > 0.30:
                return 100
            elif roi > 0.25:
                return 80
            elif roi > 0.20:
                return 40
            else:
                return -40

        elif action == 1:  # Negociar
            if 0.18 <= roi <= 0.25:
                return 50
            else:
                return 0

        else:  # Aguardar
            if roi < 0.18:
                return 30
            else:
                return -20

    def reward_payback(self, state, action, next_state):
        """Minimiza tempo de retorno."""
        payback_meses = self._calcular_payback(state)

        # Meta: payback < 36 meses
        if action == 2:  # Comprar
            if payback_meses < 24:
                return 100  # Excelente!
            elif payback_meses < 36:
                return 70
            elif payback_meses < 48:
                return 30
            else:
                return -50  # Payback muito longo

        return 0

    def reward_risco(self, state, action, next_state):
        """Penaliza projetos muito arriscados."""
        # Fatores de risco
        risco_score = 0

        # Mercado saturado?
        if state['imoveis_estoque_regiao'] > 100:
            risco_score += 30

        # Preço acima da concorrência?
        nossa_preco = state['vgv_m2']
        concorrencia = state['preco_m2_concorrencia']
        if nossa_preco > concorrencia * 1.1:
            risco_score += 20

        # Cenário pessimista?
        if state['cenario'] == 0:  # Pessimista
            risco_score += 25

        # Muita dependência de financiamento?
        if state['pode_financiar_terreno'] == 0 and state['pode_financiar_obra'] == 0:
            risco_score += 15

        # Prazo muito longo?
        prazo_total = state['prazo_obra_meses'] + state['prazo_vendas_meses']
        if prazo_total > 60:  # >5 anos
            risco_score += 20

        # Decisão baseada no risco
        if action == 2:  # Comprar
            return -risco_score
        elif action == 0:  # Aguardar
            if risco_score > 50:
                return 40  # Bom aguardar
            else:
                return 0
        else:  # Negociar
            if 30 < risco_score < 70:
                return 30  # Negociar faz sentido
            else:
                return 0

    def _calcular_tir(self, state):
        """Calcula TIR estimada do projeto."""
        # Investimento
        area_terreno = state['area_terreno_m2']
        preco_terreno = state['preco_terreno_m2']
        investimento_terreno = area_terreno * preco_terreno

        area_construida = state['area_construida_permitida_m2']
        custo_obra = state['custo_obra_m2']
        investimento_obra = area_construida * custo_obra

        investimento_total = investimento_terreno + investimento_obra

        # Receita
        vgv = area_construida * state['vgv_m2']

        # Custos de incorporação
        taxa_incorporacao = state['taxa_incorporacao']
        taxa_corretagem = state['taxa_corretagem']
        taxa_marketing = state['taxa_marketing']
        impostos = state['impostos_percentual']

        custos_total = vgv * (taxa_incorporacao + taxa_corretagem + taxa_marketing + impostos)

        # Lucro líquido
        lucro = vgv - investimento_total - custos_total

        # TIR simplificada (anualizada)
        prazo_total_meses = state['prazo_obra_meses'] + state['prazo_vendas_meses']
        prazo_anos = prazo_total_meses / 12

        if prazo_anos > 0 and investimento_total > 0:
            tir = (lucro / investimento_total) / prazo_anos
        else:
            tir = 0

        return max(-0.5, min(0.5, tir))  # Limita entre -50% e +50%

    def _calcular_roi(self, state):
        """Calcula ROI do projeto."""
        area_terreno = state['area_terreno_m2']
        preco_terreno = state['preco_terreno_m2']
        investimento_terreno = area_terreno * preco_terreno

        area_construida = state['area_construida_permitida_m2']
        custo_obra = state['custo_obra_m2']
        investimento_obra = area_construida * custo_obra

        investimento_total = investimento_terreno + investimento_obra

        vgv = area_construida * state['vgv_m2']

        custos = vgv * (state['taxa_incorporacao'] + state['taxa_corretagem'] +
                       state['taxa_marketing'] + state['impostos_percentual'])

        lucro = vgv - investimento_total - custos

        if investimento_total > 0:
            roi = lucro / investimento_total
        else:
            roi = 0

        return max(-1, min(2, roi))  # Limita entre -100% e +200%

    def _calcular_payback(self, state):
        """Calcula payback em meses."""
        # Início de retorno = quando começam as vendas
        # Normalmente durante a obra
        inicio_vendas = state['prazo_obra_meses'] * 0.3  # 30% da obra

        # Fluxo de vendas
        prazo_vendas = state['prazo_vendas_meses']

        # Payback = tempo até recuperar investimento
        # Simplificação: início vendas + metade do prazo de vendas
        payback = inicio_vendas + (prazo_vendas / 2)

        return payback


def treinar_modelo():
    """Treina o modelo de viabilidade."""
    print("="*70)
    print("TREINAMENTO: Análise de Viabilidade de Incorporação")
    print("="*70)

    problema = ViabilidadeIncorporacao()

    print("\n📊 Informações do problema:")
    info = problema.get_info()
    print(f"  Observações: {info['observation_dim']} variáveis")
    print(f"  Decisões: {info['action_space']['action']['n']} ações")
    print(f"  Objetivos: {info['n_objectives']} (TIR, ROI, Payback, Risco)")
    print(f"  Gestão de risco: CVaR (α=5%, max drawdown=30%)")

    print("\n🏋️ Treinando modelo (2 horas)...")
    modelo = brl.train(
        problema,
        algorithm='PPO',
        hours=2,
        config={'learning_rate': 3e-4}
    )

    modelo.save('./modelos/viabilidade_incorporacao.pt')
    print("\n✅ Modelo salvo!")

    return modelo


def analisar_oportunidades():
    """Analisa 3 oportunidades diferentes."""
    print("\n" + "="*70)
    print("ANÁLISE: 3 Oportunidades de Incorporação")
    print("="*70)

    modelo = brl.load('./modelos/viabilidade_incorporacao.pt')

    oportunidades = [
        {
            'nome': 'Terreno Centro - Alto Padrão',
            'dados': {
                'preco_terreno_m2': 2500,
                'area_terreno_m2': 1000,
                'area_construida_permitida_m2': 3000,
                'vgv_m2': 10000,
                'custo_obra_m2': 4500,
                'prazo_obra_meses': 24,
                'prazo_vendas_meses': 18,
                'vendas_durante_obra': 0.60,
                'vendas_pos_obra': 0.35,
                'entrada_percentual': 0.25,
                'mensais_durante_obra_perc': 0.40,
                'saldo_na_chave_perc': 0.35,
                'taxa_incorporacao': 0.10,
                'taxa_corretagem': 0.06,
                'taxa_marketing': 0.04,
                'impostos_percentual': 0.05,
                'absorcao_mercado_un_mes': 8,
                'preco_m2_concorrencia': 9500,
                'imoveis_estoque_regiao': 45,
                'taxa_juros_anual': 0.12,
                'pode_financiar_terreno': 1,
                'pode_financiar_obra': 1,
                'cenario': 1,  # Realista
                'inflacao_anual': 0.05,
                'renda_media_regiao': 12000
            }
        },
        {
            'nome': 'Terreno Subúrbio - Econômico',
            'dados': {
                'preco_terreno_m2': 800,
                'area_terreno_m2': 2500,
                'area_construida_permitida_m2': 6000,
                'vgv_m2': 4500,
                'custo_obra_m2': 2200,
                'prazo_obra_meses': 18,
                'prazo_vendas_meses': 12,
                'vendas_durante_obra': 0.50,
                'vendas_pos_obra': 0.45,
                'entrada_percentual': 0.15,
                'mensais_durante_obra_perc': 0.35,
                'saldo_na_chave_perc': 0.50,
                'taxa_incorporacao': 0.08,
                'taxa_corretagem': 0.05,
                'taxa_marketing': 0.03,
                'impostos_percentual': 0.04,
                'absorcao_mercado_un_mes': 12,
                'preco_m2_concorrencia': 4200,
                'imoveis_estoque_regiao': 80,
                'taxa_juros_anual': 0.10,
                'pode_financiar_terreno': 1,
                'pode_financiar_obra': 1,
                'cenario': 2,  # Otimista
                'inflacao_anual': 0.04,
                'renda_media_regiao': 4500
            }
        },
        {
            'nome': 'Terreno Zona Nobre - Luxo (ARRISCADO)',
            'dados': {
                'preco_terreno_m2': 4000,
                'area_terreno_m2': 800,
                'area_construida_permitida_m2': 2000,
                'vgv_m2': 14000,
                'custo_obra_m2': 5500,
                'prazo_obra_meses': 30,
                'prazo_vendas_meses': 24,
                'vendas_durante_obra': 0.35,
                'vendas_pos_obra': 0.30,
                'entrada_percentual': 0.30,
                'mensais_durante_obra_perc': 0.30,
                'saldo_na_chave_perc': 0.40,
                'taxa_incorporacao': 0.12,
                'taxa_corretagem': 0.07,
                'taxa_marketing': 0.05,
                'impostos_percentual': 0.06,
                'absorcao_mercado_un_mes': 3,
                'preco_m2_concorrencia': 13000,
                'imoveis_estoque_regiao': 120,  # Muito estoque!
                'taxa_juros_anual': 0.15,
                'pode_financiar_terreno': 0,  # Sem financiamento
                'pode_financiar_obra': 0,
                'cenario': 0,  # Pessimista
                'inflacao_anual': 0.08,
                'renda_media_regiao': 20000
            }
        }
    ]

    acoes = ["Aguardar", "Negociar", "Comprar"]

    for oport in oportunidades:
        print(f"\n{'='*70}")
        print(f"🏢 {oport['nome']}")
        print(f"{'='*70}")

        dados = oport['dados']

        # Modelo decide
        decisao = modelo.decide(dados, deterministic=True)
        acao = decisao.action

        # Calcula indicadores
        problema = ViabilidadeIncorporacao()
        tir = problema._calcular_tir(dados)
        roi = problema._calcular_roi(dados)
        payback = problema._calcular_payback(dados)

        print(f"\n📊 Indicadores Financeiros:")
        print(f"  TIR: {tir*100:.1f}% ao ano")
        print(f"  ROI: {roi*100:.1f}%")
        print(f"  Payback: {payback:.0f} meses")

        print(f"\n📐 Projeto:")
        area_const = dados['area_construida_permitida_m2']
        vgv_total = area_const * dados['vgv_m2']
        invest_terreno = dados['area_terreno_m2'] * dados['preco_terreno_m2']
        invest_obra = area_const * dados['custo_obra_m2']
        invest_total = invest_terreno + invest_obra

        print(f"  Área construída: {area_const:,.0f} m²")
        print(f"  VGV total: R$ {vgv_total:,.0f}")
        print(f"  Investimento: R$ {invest_total:,.0f}")
        print(f"  Prazo: {dados['prazo_obra_meses'] + dados['prazo_vendas_meses']:.0f} meses")

        print(f"\n✅ DECISÃO: {acoes[acao]}")

        # Justificativa
        if acao == 2:  # Comprar
            print(f"   💡 Projeto viável! Bons indicadores financeiros")
        elif acao == 1:  # Negociar
            print(f"   💡 Marginal. Negociar melhores condições")
        else:  # Aguardar
            print(f"   💡 Não viável no momento. Aguardar oportunidade melhor")

        print()


if __name__ == "__main__":
    import os
    os.makedirs('./modelos', exist_ok=True)

    print("\n💼 Análise de Viabilidade de Incorporação Imobiliária\n")

    # 1. Treina
    treinar_modelo()

    # 2. Analisa oportunidades
    analisar_oportunidades()

    print("="*70)
    print("✅ PRONTO!")
    print("   Modelo avalia viabilidade considerando:")
    print("   - TIR, ROI e Payback")
    print("   - Gestão de risco (CVaR)")
    print("   - Múltiplos cenários")
    print("="*70)
