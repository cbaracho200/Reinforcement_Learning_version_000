"""
🏠 Exemplo Básico 1: Precificação de Imóveis

PROBLEMA REAL:
Você é um corretor/imobiliária e precisa definir o preço de venda
de um imóvel para:
- Vender rápido
- Maximizar o lucro
- Ser competitivo no mercado

DECISÃO: Qual preço cobrar?

USO:
python examples/basic/01_precificacao_imoveis.py
"""

import business_rl as brl
import numpy as np


@brl.problem(name="PrecificacaoImovel")
class PrecificacaoImovel:
    """
    Problema: Definir preço de venda de um imóvel

    Observações: Características do imóvel e mercado
    Ação: Preço de venda (em R$)
    Objetivos: Maximizar lucro e velocidade de venda
    """

    obs = brl.Dict(
        # Características do imóvel
        area_m2=brl.Box(30, 500),           # Área em m²
        quartos=brl.Discrete(5),            # 0-4 quartos
        banheiros=brl.Discrete(4),          # 0-3 banheiros
        vagas_garagem=brl.Discrete(5),      # 0-4 vagas
        andar=brl.Discrete(30),             # Andar (0=térreo)
        idade_anos=brl.Box(0, 50),          # Idade do imóvel

        # Localização
        distancia_centro_km=brl.Box(0, 30),
        distancia_metro_km=brl.Box(0, 5),
        comercio_proximo=brl.Discrete(2),   # Tem comércio? 0=não, 1=sim

        # Mercado
        preco_medio_m2_regiao=brl.Box(3000, 15000),  # R$/m² na região
        imoveis_disponiveis_regiao=brl.Box(0, 100),  # Oferta
        tempo_medio_venda_dias=brl.Box(30, 365),     # Quanto tempo leva pra vender

        # Custos
        custo_aquisicao=brl.Box(100000, 5000000),    # Quanto pagou
        iptu_mensal=brl.Box(100, 5000),
        condominio_mensal=brl.Box(0, 3000)
    )

    # Decide o preço de venda
    action = brl.Box(150000, 10000000)  # Preço entre R$ 150k e R$ 10M

    # Múltiplos objetivos
    objectives = brl.Terms(
        lucro=0.60,              # 60% peso no lucro
        velocidade_venda=0.30,   # 30% peso em vender rápido
        competitividade=0.10     # 10% peso em ser competitivo
    )

    def reward_lucro(self, state, action, next_state):
        """Maximiza lucro da venda."""
        preco_venda = action
        custo_total = state['custo_aquisicao']

        # Lucro bruto
        lucro = preco_venda - custo_total

        # Normaliza para ficar entre -100 e +100
        lucro_percentual = (lucro / custo_total) * 100

        return lucro_percentual

    def reward_velocidade_venda(self, state, action, next_state):
        """Recompensa vender rápido."""
        preco_venda = action
        area = state['area_m2']
        preco_m2 = preco_venda / area

        # Compara com preço médio da região
        preco_medio_regiao = state['preco_medio_m2_regiao']
        ratio = preco_m2 / preco_medio_regiao

        # Quanto mais barato que a média, mais rápido vende
        if ratio < 0.8:  # 20% abaixo da média
            return 50  # Vende muito rápido!
        elif ratio < 0.9:  # 10% abaixo
            return 30  # Vende rápido
        elif ratio < 1.0:  # Até a média
            return 10  # Vende normal
        elif ratio < 1.1:  # 10% acima
            return -10  # Demora
        else:  # Muito acima
            return -30  # Demora muito!

    def reward_competitividade(self, state, action, next_state):
        """Mantém preço competitivo."""
        preco_venda = action
        area = state['area_m2']
        preco_m2 = preco_venda / area
        preco_medio = state['preco_medio_m2_regiao']

        # Penaliza se fugir muito da média
        diferenca = abs(preco_m2 - preco_medio) / preco_medio

        if diferenca < 0.05:  # Muito próximo
            return 20
        elif diferenca < 0.10:  # Razoável
            return 10
        elif diferenca < 0.20:  # Um pouco longe
            return 0
        else:  # Muito longe
            return -20


def treinar_modelo():
    """Treina o modelo de precificação."""
    print("="*70)
    print("TREINAMENTO: Precificação de Imóveis")
    print("="*70)

    # Cria problema
    problema = PrecificacaoImovel()

    print("\n📊 Informações do problema:")
    print(f"  Observações: {problema.get_info()['observation_dim']} variáveis")
    print(f"  Ação: Preço contínuo (R$ 150k - R$ 10M)")
    print(f"  Objetivos: {problema.get_info()['n_objectives']}")

    # Treina
    print("\n🏋️ Treinando modelo (30 minutos)...")
    modelo = brl.train(
        problema,
        algorithm='SAC',  # SAC é melhor para ações contínuas
        hours=0.5,
        config={
            'learning_rate': 3e-4,
            'batch_size': 256
        }
    )

    # Salva
    modelo.save('./modelos/precificacao_imoveis.pt')
    print("\n✅ Modelo salvo em './modelos/precificacao_imoveis.pt'")

    return modelo


def testar_modelo():
    """Testa o modelo com casos reais."""
    print("\n" + "="*70)
    print("TESTE: Precificando Imóveis Reais")
    print("="*70)

    # Carrega modelo
    modelo = brl.load('./modelos/precificacao_imoveis.pt')

    # Casos de teste
    casos = [
        {
            'nome': 'Apartamento Compacto - Centro',
            'imovel': {
                'area_m2': 45,
                'quartos': 1,
                'banheiros': 1,
                'vagas_garagem': 1,
                'andar': 8,
                'idade_anos': 5,
                'distancia_centro_km': 2,
                'distancia_metro_km': 0.5,
                'comercio_proximo': 1,
                'preco_medio_m2_regiao': 8000,
                'imoveis_disponiveis_regiao': 45,
                'tempo_medio_venda_dias': 90,
                'custo_aquisicao': 250000,
                'iptu_mensal': 200,
                'condominio_mensal': 500
            }
        },
        {
            'nome': 'Casa Grande - Subúrbio',
            'imovel': {
                'area_m2': 200,
                'quartos': 3,
                'banheiros': 2,
                'vagas_garagem': 2,
                'andar': 0,  # Casa térrea
                'idade_anos': 15,
                'distancia_centro_km': 15,
                'distancia_metro_km': 3,
                'comercio_proximo': 1,
                'preco_medio_m2_regiao': 4500,
                'imoveis_disponiveis_regiao': 20,
                'tempo_medio_venda_dias': 120,
                'custo_aquisicao': 650000,
                'iptu_mensal': 800,
                'condominio_mensal': 0
            }
        },
        {
            'nome': 'Cobertura Luxo - Zona Sul',
            'imovel': {
                'area_m2': 180,
                'quartos': 3,
                'banheiros': 3,
                'vagas_garagem': 3,
                'andar': 20,
                'idade_anos': 2,
                'distancia_centro_km': 8,
                'distancia_metro_km': 0.3,
                'comercio_proximo': 1,
                'preco_medio_m2_regiao': 12000,
                'imoveis_disponiveis_regiao': 15,
                'tempo_medio_venda_dias': 180,
                'custo_aquisicao': 1800000,
                'iptu_mensal': 1500,
                'condominio_mensal': 1200
            }
        }
    ]

    print("\n📋 Analisando 3 imóveis diferentes:\n")

    for caso in casos:
        print(f"{'='*70}")
        print(f"📍 {caso['nome']}")
        print(f"{'='*70}")

        imovel = caso['imovel']

        # Modelo decide preço
        decisao = modelo.decide(imovel, deterministic=True)
        preco_sugerido = decisao.action

        # Calcula métricas
        area = imovel['area_m2']
        preco_m2 = preco_sugerido / area
        preco_medio_regiao = imovel['preco_medio_m2_regiao']
        custo = imovel['custo_aquisicao']
        lucro = preco_sugerido - custo
        lucro_percentual = (lucro / custo) * 100

        print(f"\n📐 Características:")
        print(f"  Área: {area:.0f} m²")
        print(f"  Quartos: {imovel['quartos']} | Banheiros: {imovel['banheiros']} | Vagas: {imovel['vagas_garagem']}")
        print(f"  Idade: {imovel['idade_anos']:.0f} anos")
        print(f"  Distância centro: {imovel['distancia_centro_km']:.1f} km")

        print(f"\n💰 Análise Financeira:")
        print(f"  Custo de aquisição: R$ {custo:,.2f}")
        print(f"  Preço médio/m² região: R$ {preco_medio_regiao:,.2f}")

        print(f"\n✅ PREÇO SUGERIDO: R$ {preco_sugerido:,.2f}")
        print(f"  Preço/m²: R$ {preco_m2:,.2f}")
        print(f"  vs Média: {(preco_m2/preco_medio_regiao - 1)*100:+.1f}%")
        print(f"  Lucro esperado: R$ {lucro:,.2f} ({lucro_percentual:+.1f}%)")

        # Estimativa de tempo de venda
        ratio = preco_m2 / preco_medio_regiao
        if ratio < 0.9:
            tempo_estimado = "30-60 dias (rápido!)"
        elif ratio < 1.0:
            tempo_estimado = "60-90 dias (normal)"
        elif ratio < 1.1:
            tempo_estimado = "90-120 dias (lento)"
        else:
            tempo_estimado = "120+ dias (muito lento)"

        print(f"  Tempo estimado venda: {tempo_estimado}")
        print()


if __name__ == "__main__":
    import os
    os.makedirs('./modelos', exist_ok=True)

    print("\n🏠 Precificação Inteligente de Imóveis\n")

    # 1. Treina
    treinar_modelo()

    # 2. Testa
    testar_modelo()

    print("="*70)
    print("✅ PRONTO!")
    print("   Agora você tem um modelo que precifica imóveis automaticamente!")
    print("   Adaptável para sua carteira de imóveis.")
    print("="*70)
