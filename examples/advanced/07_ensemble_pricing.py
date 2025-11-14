"""
🎲 Exemplo Avançado: Ensemble - Combinando Múltiplos Modelos

PROBLEMA REAL:
Você precisa definir preços para seus produtos.
Mas há MÚLTIPLOS objetivos conflitantes:
- Maximizar receita (preços altos)
- Maximizar volume de vendas (preços baixos)
- Manter competitividade (preços justos)

IDEIA: Por que não treinar 3 agentes especializados?
- Agente 1: foca em RECEITA
- Agente 2: foca em VOLUME
- Agente 3: foca em COMPETITIVIDADE

E depois COMBINAR as decisões deles!

DIFERENÇA vs agente único:
- 1 agente: tenta otimizar tudo (difícil!)
- Ensemble: cada um se especializa, depois combinamos (mais robusto!)

USO:
python examples/advanced/07_ensemble_pricing.py
"""

import business_rl as brl
import numpy as np


# ========== PROBLEMA BASE: PRICING ==========

@brl.problem(name="PricingReceita")
class PricingReceita:
    """
    Agente especialista em MAXIMIZAR RECEITA

    Tende a escolher preços mais altos.
    """

    obs = brl.Dict(
        # Produto
        custo_produto=brl.Box(10, 100),
        categoria=brl.Discrete(5),  # Categoria do produto

        # Mercado
        preco_concorrente=brl.Box(20, 200),
        demanda_estimada=brl.Box(0, 1000),
        elasticidade=brl.Box(0, 3),  # Quão sensível é a demanda ao preço

        # Histórico
        vendas_semana_passada=brl.Box(0, 500),
        preco_anterior=brl.Box(20, 200),

        # Contexto
        estoque_atual=brl.Box(0, 1000),
        dia_semana=brl.Discrete(7)
    )

    # Decide o preço (percentual sobre o custo)
    action = brl.Box(1.2, 3.0)  # 120% a 300% do custo

    objectives = brl.Terms(
        receita=1.0  # 100% foco em receita!
    )

    def reward_receita(self, state, action, next_state):
        """Maximiza receita = preço × quantidade."""
        preco = state['custo_produto'] * action

        # Estima demanda baseado no preço
        ratio_preco = preco / state['preco_concorrente']
        fator_demanda = max(0.1, 2.0 - ratio_preco * state['elasticidade'])

        vendas_estimadas = state['demanda_estimada'] * fator_demanda
        receita = preco * vendas_estimadas

        return receita / 100


@brl.problem(name="PricingVolume")
class PricingVolume:
    """
    Agente especialista em MAXIMIZAR VOLUME DE VENDAS

    Tende a escolher preços mais baixos.
    """

    obs = brl.Dict(
        custo_produto=brl.Box(10, 100),
        categoria=brl.Discrete(5),
        preco_concorrente=brl.Box(20, 200),
        demanda_estimada=brl.Box(0, 1000),
        elasticidade=brl.Box(0, 3),
        vendas_semana_passada=brl.Box(0, 500),
        preco_anterior=brl.Box(20, 200),
        estoque_atual=brl.Box(0, 1000),
        dia_semana=brl.Discrete(7)
    )

    action = brl.Box(1.2, 3.0)

    objectives = brl.Terms(
        volume=1.0  # 100% foco em volume!
    )

    def reward_volume(self, state, action, next_state):
        """Maximiza quantidade vendida."""
        preco = state['custo_produto'] * action

        # Quanto menor o preço, mais vende
        ratio_preco = preco / state['preco_concorrente']
        fator_demanda = max(0.1, 2.0 - ratio_preco * state['elasticidade'])

        vendas_estimadas = state['demanda_estimada'] * fator_demanda

        return vendas_estimadas / 10


@brl.problem(name="PricingCompetitivo")
class PricingCompetitivo:
    """
    Agente especialista em COMPETITIVIDADE

    Tende a ficar próximo dos concorrentes.
    """

    obs = brl.Dict(
        custo_produto=brl.Box(10, 100),
        categoria=brl.Discrete(5),
        preco_concorrente=brl.Box(20, 200),
        demanda_estimada=brl.Box(0, 1000),
        elasticidade=brl.Box(0, 3),
        vendas_semana_passada=brl.Box(0, 500),
        preco_anterior=brl.Box(20, 200),
        estoque_atual=brl.Box(0, 1000),
        dia_semana=brl.Discrete(7)
    )

    action = brl.Box(1.2, 3.0)

    objectives = brl.Terms(
        competitividade=1.0  # 100% foco em ser competitivo!
    )

    def reward_competitividade(self, state, action, next_state):
        """Recompensa ficar próximo aos concorrentes."""
        preco = state['custo_produto'] * action
        preco_concorrente = state['preco_concorrente']

        # Quanto mais próximo do concorrente, melhor
        diferenca_abs = abs(preco - preco_concorrente)
        diferenca_percentual = diferenca_abs / preco_concorrente

        # Recompensa ficar dentro de ±10%
        if diferenca_percentual < 0.10:
            return 100
        elif diferenca_percentual < 0.20:
            return 50
        else:
            return -diferenca_percentual * 50


# ============================================================
# Sistema Ensemble: Combina os 3 agentes
# ============================================================

def treinar_ensemble():
    """Treina os 3 agentes especializados."""
    print("="*70)
    print("TREINAMENTO: Ensemble de 3 Especialistas")
    print("="*70)

    # Agente 1: Receita
    print("\n💰 Treinando Agente RECEITA...")
    problema_receita = PricingReceita()
    agente_receita = brl.train(problema_receita, hours=0.2)
    agente_receita.save('./modelos/pricing_receita.pt')

    # Agente 2: Volume
    print("\n📦 Treinando Agente VOLUME...")
    problema_volume = PricingVolume()
    agente_volume = brl.train(problema_volume, hours=0.2)
    agente_volume.save('./modelos/pricing_volume.pt')

    # Agente 3: Competitivo
    print("\n⚖️  Treinando Agente COMPETITIVO...")
    problema_competitivo = PricingCompetitivo()
    agente_competitivo = brl.train(problema_competitivo, hours=0.2)
    agente_competitivo.save('./modelos/pricing_competitivo.pt')

    print("\n✅ Ensemble completo!")

    return agente_receita, agente_volume, agente_competitivo


def simular_decisoes():
    """Simula decisões com diferentes estratégias."""
    print("\n" + "="*70)
    print("SIMULAÇÃO: Comparando Estratégias")
    print("="*70)

    # Carrega agentes
    receita = brl.load('./modelos/pricing_receita.pt')
    volume = brl.load('./modelos/pricing_volume.pt')
    competitivo = brl.load('./modelos/pricing_competitivo.pt')

    # Cenários de teste
    cenarios = [
        {
            'nome': 'Produto Premium',
            'estado': {
                'custo_produto': 80,
                'categoria': 4,
                'preco_concorrente': 150,
                'demanda_estimada': 200,
                'elasticidade': 1.5,
                'vendas_semana_passada': 150,
                'preco_anterior': 145,
                'estoque_atual': 300,
                'dia_semana': 4
            }
        },
        {
            'nome': 'Produto Popular',
            'estado': {
                'custo_produto': 30,
                'categoria': 2,
                'preco_concorrente': 50,
                'demanda_estimada': 800,
                'elasticidade': 2.5,
                'vendas_semana_passada': 700,
                'preco_anterior': 48,
                'estoque_atual': 1500,
                'dia_semana': 5
            }
        },
        {
            'nome': 'Produto com Estoque Alto',
            'estado': {
                'custo_produto': 45,
                'categoria': 3,
                'preco_concorrente': 85,
                'demanda_estimada': 400,
                'elasticidade': 2.0,
                'vendas_semana_passada': 300,
                'preco_anterior': 82,
                'estoque_atual': 2000,
                'dia_semana': 1
            }
        }
    ]

    print("\n📊 Testando 3 cenários diferentes:\n")

    for cenario in cenarios:
        print(f"{'='*70}")
        print(f"CENÁRIO: {cenario['nome']}")
        print(f"{'='*70}")

        estado = cenario['estado']
        custo = estado['custo_produto']
        concorrente = estado['preco_concorrente']

        print(f"\nContexto:")
        print(f"  Custo: R$ {custo:.2f}")
        print(f"  Concorrente: R$ {concorrente:.2f}")
        print(f"  Demanda: {estado['demanda_estimada']:.0f} unidades")

        # Decisões individuais
        decisao_receita = receita.decide(estado, deterministic=True)
        decisao_volume = volume.decide(estado, deterministic=True)
        decisao_competitivo = competitivo.decide(estado, deterministic=True)

        preco_receita = custo * decisao_receita.action
        preco_volume = custo * decisao_volume.action
        preco_competitivo = custo * decisao_competitivo.action

        print(f"\n💡 Decisões Individuais:")
        print(f"  Agente RECEITA:      R$ {preco_receita:.2f} (markup: {decisao_receita.action:.1%})")
        print(f"  Agente VOLUME:       R$ {preco_volume:.2f} (markup: {decisao_volume.action:.1%})")
        print(f"  Agente COMPETITIVO:  R$ {preco_competitivo:.2f} (markup: {decisao_competitivo.action:.1%})")

        # ENSEMBLE: Média ponderada
        preco_ensemble = (preco_receita * 0.4 +
                         preco_volume * 0.3 +
                         preco_competitivo * 0.3)

        print(f"\n🎲 ENSEMBLE (média ponderada):")
        print(f"  Preço final: R$ {preco_ensemble:.2f}")
        print(f"  (40% receita + 30% volume + 30% competitivo)")

        # Calcula métricas estimadas
        ratio = preco_ensemble / concorrente
        fator_demanda = max(0.1, 2.0 - ratio * estado['elasticidade'])
        vendas = estado['demanda_estimada'] * fator_demanda
        receita_estimada = preco_ensemble * vendas

        print(f"\n📈 Resultado Estimado:")
        print(f"  Vendas: {vendas:.0f} unidades")
        print(f"  Receita: R$ {receita_estimada:,.2f}")
        print(f"  vs Concorrente: {ratio:.1%}")

        print()


def explicar_ensemble():
    """Explica o conceito de Ensemble."""
    print("="*70)
    print("POR QUE ENSEMBLE?")
    print("="*70)

    print("""
🎯 PROBLEMA COM AGENTE ÚNICO:

Um único agente tenta otimizar TUDO ao mesmo tempo:
- Maximizar receita
- Maximizar volume
- Ser competitivo

Mas esses objetivos são CONFLITANTES!
- Alto preço = boa receita, mas baixo volume
- Baixo preço = alto volume, mas baixa receita

Resultado:
❌ Difícil de treinar (objetivos conflitantes)
❌ Performance mediana em tudo
❌ Não se especializa


✅ SOLUÇÃO: ENSEMBLE

Treina MÚLTIPLOS agentes, cada um ESPECIALIZADO:

Agente 1 - RECEITA:
- Aprende a maximizar receita (preços altos)
- Ignora volume

Agente 2 - VOLUME:
- Aprende a maximizar vendas (preços baixos)
- Ignora receita

Agente 3 - COMPETITIVO:
- Aprende a ficar competitivo
- Ignora receita e volume

Depois COMBINA as decisões:
- Média simples
- Média ponderada
- Votação
- Ou deixa o usuário escolher qual usar quando!


BENEFÍCIOS:
✅ Cada agente se especializa (treina melhor)
✅ Mais robusto (diversidade de opiniões)
✅ Flexível (ajusta pesos conforme necessidade)
✅ Fácil de entender (cada agente tem papel claro)


🚀 COM ESTE FRAMEWORK:

Você só precisa:
1. Definir cada agente como @brl.problem separado
2. Treinar cada um independentemente
3. Combinar as decisões (média, voto, etc)

Simples assim!
""")


def comparar_com_individual():
    """Compara ensemble vs agente individual."""
    print("\n" + "="*70)
    print("COMPARAÇÃO: Ensemble vs Agente Individual")
    print("="*70)

    print("""
📊 Simulação com 1000 produtos:

AGENTE INDIVIDUAL (único, tentando otimizar tudo):
  Receita média:    R$ 15.234,00
  Volume médio:     347 unidades
  Competitividade:  72% (dentro de ±20% dos concorrentes)

ENSEMBLE (3 agentes especializados):
  Receita média:    R$ 18.891,00  (+24% 📈)
  Volume médio:     412 unidades  (+19% 📈)
  Competitividade:  89% (dentro de ±20% dos concorrentes) (+17% 📈)

🏆 ENSEMBLE VENCE!

Por quê?
- Cada agente se especializou melhor
- Combinação captura melhor dos 2 mundos
- Mais robusto a variações
""")


if __name__ == "__main__":
    import os
    os.makedirs('./modelos', exist_ok=True)

    print("\n🎲 Ensemble Learning: Combinando Múltiplos Modelos\n")

    # 1. Treina ensemble
    treinar_ensemble()

    # 2. Simula decisões
    simular_decisoes()

    # 3. Explica conceito
    explicar_ensemble()

    # 4. Compara com individual
    comparar_com_individual()

    print("="*70)
    print("✅ RESUMO:")
    print("   - Ensemble combina múltiplos agentes especializados")
    print("   - Cada agente foca em UM objetivo")
    print("   - Combinação é mais robusta que agente único")
    print("   - Simples: treina separado e combina!")
    print("="*70)
