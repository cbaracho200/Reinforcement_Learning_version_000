"""
🤝 Exemplo Avançado: Multi-Agent - Cooperação Simples

PROBLEMA REAL:
Imagine uma loja com 2 departamentos:
- COMPRAS: Decide quanto comprar dos fornecedores
- VENDAS: Decide preço e promoções

Eles precisam COOPERAR:
- Compras não pode comprar demais (estoque parado)
- Vendas não pode vender rápido demais (ruptura)

DESAFIO:
Treinar 2 agentes que aprendem a trabalhar juntos!

DIFERENÇA vs agente único:
- 1 agente: toma todas as decisões
- Multi-agent: cada um se especializa, mas cooperam

USO:
python examples/advanced/05_multi_agent_cooperation.py
"""

import business_rl as brl
import numpy as np


# =========== AGENTE 1: COMPRAS ===========

@brl.problem(name="AgenteCompras")
class AgenteCompras:
    """
    Agente que decide QUANTO COMPRAR dos fornecedores

    Vê: estoque atual, demanda prevista, orçamento
    Decide: quantidade a comprar
    Objetivo: manter estoque adequado com menor custo
    """

    obs = brl.Dict(
        # Estado do estoque
        estoque_atual=brl.Box(0, 1000),
        estoque_minimo_seguranca=brl.Box(50, 200),

        # Demanda
        demanda_semana_passada=brl.Box(0, 500),
        demanda_prevista=brl.Box(0, 500),

        # Financeiro
        orcamento_disponivel=brl.Box(0, 50000),
        custo_unitario=brl.Box(10, 50),

        # Comunicação do agente de Vendas
        preco_venda_planejado=brl.Box(20, 200),  # Quanto Vendas vai cobrar
        promocao_planejada=brl.Discrete(2)        # Vendas vai fazer promoção?
    )

    action = brl.Dict(
        quantidade_compra=brl.Box(0, 500),  # Quanto comprar

        # Comunica para Vendas
        quantidade_disponivel=brl.Box(0, 1000)  # Quanto terá disponível
    )

    objectives = brl.Terms(
        custo=0.40,              # Minimizar custo de compra
        disponibilidade=0.35,    # Manter estoque
        coordenacao=0.25         # Coordenar com Vendas
    )

    def reward_custo(self, state, action, next_state):
        """Minimiza custo de compra."""
        custo = action['quantidade_compra'] * state['custo_unitario']
        return -custo / 100  # Negativo para minimizar

    def reward_disponibilidade(self, state, action, next_state):
        """Mantém estoque adequado."""
        estoque_futuro = (state['estoque_atual'] +
                         action['quantidade_compra'])

        # Muito estoque = ruim (capital parado)
        if estoque_futuro > 800:
            return -50

        # Pouco estoque = ruim (risco ruptura)
        if estoque_futuro < state['estoque_minimo_seguranca']:
            return -100

        # Estoque adequado = bom!
        return 50

    def reward_coordenacao(self, state, action, next_state):
        """Coordena com Vendas."""
        # Se Vendas planeja promoção, Compras deve ter estoque!
        if state['promocao_planejada'] == 1:  # Promoção
            estoque_futuro = (state['estoque_atual'] +
                            action['quantidade_compra'])

            # Precisa ter estoque extra para promoção
            if estoque_futuro > state['demanda_prevista'] * 1.5:
                return 100  # Boa coordenação!
            else:
                return -50  # Vai faltar produto na promoção

        return 0


# =========== AGENTE 2: VENDAS ===========

@brl.problem(name="AgenteVendas")
class AgenteVendas:
    """
    Agente que decide PREÇO e PROMOÇÕES

    Vê: estoque, demanda, concorrentes
    Decide: preço e se faz promoção
    Objetivo: maximizar receita sem causar ruptura
    """

    obs = brl.Dict(
        # Mercado
        demanda_atual=brl.Box(0, 500),
        preco_concorrente=brl.Box(20, 200),

        # Comunicação do agente de Compras
        estoque_disponivel=brl.Box(0, 1000),  # Quanto Compras tem
        quantidade_chegando=brl.Box(0, 500),  # Quanto Compras está comprando

        # Contexto
        dia_semana=brl.Discrete(7),
        fim_de_mes=brl.Discrete(2)
    )

    action = brl.Dict(
        preco=brl.Box(20, 200),        # Preço a cobrar
        fazer_promocao=brl.Discrete(2),  # Fazer promoção?

        # Comunica para Compras
        demanda_esperada=brl.Box(0, 500)  # Quanto espera vender
    )

    objectives = brl.Terms(
        receita=0.50,         # Maximizar receita
        disponibilidade=0.30,  # Não causar ruptura
        coordenacao=0.20      # Coordenar com Compras
    )

    def reward_receita(self, state, action, next_state):
        """Maximiza receita."""
        # Estima vendas baseado no preço
        ratio_preco = action['preco'] / state['preco_concorrente']

        # Quanto mais caro, menos vende (elasticidade)
        fator_preco = 2.0 - ratio_preco  # Simplificado

        # Promoção aumenta vendas
        fator_promocao = 1.5 if action['fazer_promocao'] == 1 else 1.0

        vendas_estimadas = (state['demanda_atual'] *
                           fator_preco *
                           fator_promocao)

        # Limita pelo estoque
        vendas_reais = min(vendas_estimadas, state['estoque_disponivel'])

        receita = vendas_reais * action['preco']
        return receita / 100

    def reward_disponibilidade(self, state, action, next_state):
        """Não causa ruptura."""
        vendas_estimadas = action['demanda_esperada']

        # Se estoque é insuficiente, penaliza
        if vendas_estimadas > state['estoque_disponivel']:
            falta = vendas_estimadas - state['estoque_disponivel']
            return -falta * 2

        return 0

    def reward_coordenacao(self, state, action, next_state):
        """Coordena com Compras."""
        # Não fazer promoção se estoque está baixo
        if action['fazer_promocao'] == 1:
            if state['estoque_disponivel'] < 300:
                return -100  # Má coordenação!
            else:
                return 50   # Boa coordenação!

        return 0


# ============================================================
# Sistema Multi-Agent: Treina e usa os 2 agentes juntos
# ============================================================

def treinar_agentes():
    """Treina os 2 agentes (pode ser em paralelo!)."""
    print("="*70)
    print("TREINAMENTO: 2 Agentes Cooperando")
    print("="*70)

    # Treina Agente de Compras
    print("\n🛒 Treinando Agente de COMPRAS...")
    problema_compras = AgenteCompras()
    agente_compras = brl.train(problema_compras, hours=0.25)  # 15min
    agente_compras.save('./modelos/agente_compras.pt')

    # Treina Agente de Vendas
    print("💰 Treinando Agente de VENDAS...")
    problema_vendas = AgenteVendas()
    agente_vendas = brl.train(problema_vendas, hours=0.25)  # 15min
    agente_vendas.save('./modelos/agente_vendas.pt')

    print("\n✅ Ambos os agentes treinados!")

    return agente_compras, agente_vendas


def simular_cooperacao():
    """Simula os 2 agentes cooperando."""
    print("\n" + "="*70)
    print("SIMULAÇÃO: Agentes Cooperando")
    print("="*70)

    # Carrega agentes
    compras = brl.load('./modelos/agente_compras.pt')
    vendas = brl.load('./modelos/agente_vendas.pt')

    # Estado inicial
    estoque = 300
    orcamento = 20000

    print("\n📊 Simulando 5 dias de operação:\n")

    for dia in range(1, 6):
        print(f"📅 DIA {dia}")
        print("-" * 70)

        # 1. VENDAS decide primeiro (preço e promoção)
        estado_vendas = {
            'demanda_atual': np.random.randint(80, 120),
            'preco_concorrente': 100,
            'estoque_disponivel': estoque,
            'quantidade_chegando': 0,
            'dia_semana': dia % 7,
            'fim_de_mes': 0
        }

        decisao_vendas = vendas.decide(estado_vendas, deterministic=True)

        print(f"💰 VENDAS decidiu:")
        print(f"   Preço: R$ {decisao_vendas.action['preco']:.2f}")
        print(f"   Promoção: {'SIM' if decisao_vendas.action['fazer_promocao'] == 1 else 'NÃO'}")

        # 2. COMPRAS decide baseado no que VENDAS planejou
        estado_compras = {
            'estoque_atual': estoque,
            'estoque_minimo_seguranca': 100,
            'demanda_semana_passada': 100,
            'demanda_prevista': decisao_vendas.action['demanda_esperada'],
            'orcamento_disponivel': orcamento,
            'custo_unitario': 40,
            'preco_venda_planejado': decisao_vendas.action['preco'],
            'promocao_planejada': decisao_vendas.action['fazer_promocao']
        }

        decisao_compras = compras.decide(estado_compras, deterministic=True)

        print(f"🛒 COMPRAS decidiu:")
        print(f"   Comprar: {decisao_compras.action['quantidade_compra']:.0f} unidades")

        # Atualiza estado (simplificado)
        vendas_dia = min(estado_vendas['demanda_atual'], estoque)
        estoque = estoque - vendas_dia + decisao_compras.action['quantidade_compra']
        orcamento -= decisao_compras.action['quantidade_compra'] * 40

        print(f"\n📈 Resultado do dia:")
        print(f"   Vendeu: {vendas_dia:.0f} unidades")
        print(f"   Estoque final: {estoque:.0f}")
        print(f"   Orçamento restante: R$ {orcamento:,.2f}\n")


def exemplo_coordenacao():
    """Mostra importância da coordenação."""
    print("="*70)
    print("IMPORTÂNCIA DA COORDENAÇÃO")
    print("="*70)

    print("""
🎯 POR QUE MULTI-AGENT?

SEM Coordenação (cada um por si):
- Vendas faz promoção → Estoque acaba → Clientes insatisfeitos
- Compras compra demais → Vendas não consegue vender → Prejuízo

COM Coordenação (cooperando):
- Vendas avisa que vai fazer promoção
- Compras garante estoque extra
- Resultado: Promoção bem-sucedida!

✅ O framework aprende automaticamente a coordenar!
   Você só define:
   1. O que cada agente vê
   2. O que cada agente decide
   3. Como eles se comunicam (campos compartilhados)

   O resto é automático!
""")


if __name__ == "__main__":
    import os
    os.makedirs('./modelos', exist_ok=True)

    print("\n🤝 Multi-Agent: Cooperação entre Agentes\n")

    # 1. Treina os 2 agentes
    treinar_agentes()

    # 2. Simula cooperação
    simular_cooperacao()

    # 3. Explica importância
    exemplo_coordenacao()

    print("="*70)
    print("✅ RESUMO:")
    print("   - Multi-Agent: múltiplos agentes especializados cooperando")
    print("   - Cada agente aprende sua parte")
    print("   - Coordenação emerge automaticamente do treinamento")
    print("   - Mais escalável que um único agente gigante")
    print("="*70)
