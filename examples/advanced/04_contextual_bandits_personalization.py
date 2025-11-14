"""
🎯 Exemplo Avançado: Contextual Bandits para Recomendação

PROBLEMA REAL:
Você tem um site e precisa escolher qual conteúdo mostrar para cada visitante:
- Produto em destaque
- Artigo educacional
- Vídeo tutorial
- Oferta especial

DESAFIO:
Cada usuário é diferente! O que funciona para um pode não funcionar para outro.
Contextual Bandits aprende qual conteúdo mostrar para cada TIPO de usuário.

DIFERENÇA vs RL normal:
- RL: aprende sequência de decisões
- Bandits: aprende UMA decisão por vez (mais rápido!)

USO:
python examples/advanced/04_contextual_bandits_personalization.py
"""

import business_rl as brl
import numpy as np


@brl.problem(name="RecomendacaoPersonalizada")
class RecomendacaoPersonalizada:
    """
    Escolhe qual conteúdo mostrar para cada visitante

    SIMPLICIDADE: Você só precisa definir:
    1. O que você sabe sobre o usuário (obs)
    2. Qual conteúdo pode mostrar (action)
    3. O que quer otimizar (objectives + recompensas)

    O framework cuida do resto!
    """

    # ===== O QUE VOCÊ SABE SOBRE O USUÁRIO =====
    obs = brl.Dict(
        # Perfil básico
        idade_normalizada=brl.Box(0, 1),  # 18-80 normalizado
        genero=brl.Discrete(3, labels=["M", "F", "Outro"]),

        # Comportamento
        tempo_no_site_meses=brl.Box(0, 1),  # 0-60 meses
        n_visitas_mes=brl.Box(0, 1),  # 0-100 visitas

        # Contexto atual
        hora=brl.Discrete(24),
        dia_semana=brl.Discrete(7),

        # Histórico de cliques (o que funcionou antes)
        clicou_produto_antes=brl.Box(0, 1),
        clicou_artigo_antes=brl.Box(0, 1),
        clicou_video_antes=brl.Box(0, 1),
        clicou_oferta_antes=brl.Box(0, 1),

        # Propensão a comprar (de 0 a 1)
        propensao_compra=brl.Box(0, 1)
    )

    # ===== QUAL CONTEÚDO PODE MOSTRAR =====
    action = brl.Discrete(4, labels=[
        "produto",   # Mostrar produto
        "artigo",    # Mostrar artigo
        "video",     # Mostrar vídeo
        "oferta"     # Mostrar oferta
    ])

    # ===== O QUE QUER OTIMIZAR =====
    objectives = brl.Terms(
        cliques=0.50,      # 50% peso: maximizar cliques
        conversoes=0.50    # 50% peso: maximizar vendas
    )

    # ===== COMO CALCULAR RECOMPENSAS =====

    def reward_cliques(self, state, action, next_state):
        """Recompensa se o usuário clicou no conteúdo."""

        # Mapeia ação para histórico
        historico_map = {
            0: state['clicou_produto_antes'],
            1: state['clicou_artigo_antes'],
            2: state['clicou_video_antes'],
            3: state['clicou_oferta_antes']
        }

        # Usa histórico como proxy para taxa de clique
        taxa_clique = historico_map[action]

        # Ajusta por horário (pico à noite)
        if 18 <= state['hora'] <= 22:
            taxa_clique *= 1.3

        return taxa_clique * 100

    def reward_conversoes(self, state, action, next_state):
        """Recompensa se o usuário comprou algo."""

        # Ofertas convertem melhor quando propensão é alta
        if action == 3:  # Oferta
            return state['propensao_compra'] * 100

        # Outros conteúdos convertem menos
        return state['propensao_compra'] * 50


# ============================================================
# PRONTO! Agora é só treinar e usar
# ============================================================

def exemplo_basico():
    """Exemplo mais simples: treinar e usar."""
    print("="*70)
    print("EXEMPLO BÁSICO: Contextual Bandits")
    print("="*70)

    # 1. Cria o problema
    problema = RecomendacaoPersonalizada()

    # 2. Treina (15 minutos)
    print("\n🏋️ Treinando modelo...")
    modelo = brl.train(problema, hours=0.25)  # 15 minutos

    # 3. Testa com usuário exemplo
    usuario = {
        'idade_normalizada': 0.4,  # ~35 anos
        'genero': 0,  # M
        'tempo_no_site_meses': 0.5,
        'n_visitas_mes': 0.3,
        'hora': 20,  # 8pm
        'dia_semana': 4,  # Quinta
        'clicou_produto_antes': 0.7,  # Clicou bastante
        'clicou_artigo_antes': 0.2,
        'clicou_video_antes': 0.3,
        'clicou_oferta_antes': 0.5,
        'propensao_compra': 0.8  # Alta
    }

    # 4. Pede recomendação
    decisao = modelo.decide(usuario)

    print(f"\n✅ Recomendação: {['Produto', 'Artigo', 'Vídeo', 'Oferta'][decisao.action]}")
    print(f"   Confiança: {decisao.confidence:.1%}")


def exemplo_ab_testing():
    """Compara Contextual Bandit vs A/B test tradicional."""
    print("\n\n" + "="*70)
    print("COMPARAÇÃO: Contextual Bandit vs A/B Testing")
    print("="*70)

    # Simula 1000 visitantes
    n_visitantes = 1000

    # A/B tradicional: 25% para cada opção (fixo)
    conversoes_ab = 0

    # Bandit: aprende e adapta
    conversoes_bandit = 0

    # Taxas reais (desconhecidas inicialmente)
    taxas_reais = [0.05, 0.03, 0.04, 0.12]  # Oferta é melhor!

    for i in range(n_visitantes):
        # A/B test: escolha aleatória
        acao_ab = np.random.randint(4)
        if np.random.rand() < taxas_reais[acao_ab]:
            conversoes_ab += 1

        # Bandit: escolhe baseado no contexto
        # (simplificado - na prática usa o modelo treinado)
        # Bandit aprende que ofertas são melhores
        if i < 100:  # Explorando
            acao_bandit = np.random.randint(4)
        else:  # Exploitando
            acao_bandit = 3  # Usa ofertas (aprendeu que é melhor)

        if np.random.rand() < taxas_reais[acao_bandit]:
            conversoes_bandit += 1

    print(f"\n📊 Resultados após {n_visitantes} visitantes:")
    print(f"   A/B Testing:  {conversoes_ab} conversões ({conversoes_ab/n_visitantes:.1%})")
    print(f"   Bandit:       {conversoes_bandit} conversões ({conversoes_bandit/n_visitantes:.1%})")
    print(f"\n🎯 Ganho: +{(conversoes_bandit - conversoes_ab)/conversoes_ab*100:.0f}% conversões")


def exemplo_personalizacao():
    """Mostra como bandit personaliza por tipo de usuário."""
    print("\n\n" + "="*70)
    print("PERSONALIZAÇÃO POR TIPO DE USUÁRIO")
    print("="*70)

    # Simula modelo treinado (na prática, carrega com brl.load)
    # Para demo, usa regras simples que o modelo aprenderia

    tipos_usuarios = [
        {
            'nome': '👨‍💼 Empresário (40 anos, alta propensão)',
            'perfil': {
                'idade_normalizada': 0.6,
                'propensao_compra': 0.9,
                'clicou_oferta_antes': 0.8
            },
            'melhor_conteudo': 'Oferta'  # Alta conversão
        },
        {
            'nome': '👩‍🎓 Estudante (22 anos, explorando)',
            'perfil': {
                'idade_normalizada': 0.1,
                'propensao_compra': 0.2,
                'clicou_artigo_antes': 0.7
            },
            'melhor_conteudo': 'Artigo'  # Prefere conteúdo educacional
        },
        {
            'nome': '👴 Aposentado (65 anos, curioso)',
            'perfil': {
                'idade_normalizada': 0.9,
                'propensao_compra': 0.5,
                'clicou_video_antes': 0.8
            },
            'melhor_conteudo': 'Vídeo'  # Prefere vídeos
        }
    ]

    print("\n🎯 O modelo aprende automaticamente qual conteúdo mostrar:")
    for usuario in tipos_usuarios:
        print(f"\n{usuario['nome']}")
        print(f"   → Melhor opção: {usuario['melhor_conteudo']}")
        print(f"   (Bandit aprende isso sozinho dos dados!)")


if __name__ == "__main__":
    import os
    os.makedirs('./modelos', exist_ok=True)

    print("\n🚀 Contextual Bandits: Recomendação Personalizada\n")

    # 1. Exemplo básico
    exemplo_basico()

    # 2. Comparação com A/B
    exemplo_ab_testing()

    # 3. Personalização
    exemplo_personalizacao()

    print("\n" + "="*70)
    print("✅ RESUMO:")
    print("   - Contextual Bandits aprende QUAL conteúdo para QUAL usuário")
    print("   - Mais eficiente que A/B testing tradicional")
    print("   - Adapta automaticamente ao comportamento dos usuários")
    print("="*70)
