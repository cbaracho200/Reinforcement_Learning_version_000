"""
🏢 Exemplo Avançado: Hierarchical RL - Gerenciamento de Restaurante

PROBLEMA REAL:
Gerenciar um restaurante tem 2 níveis de decisão:

NÍVEL ESTRATÉGICO (Gerente):
- Decide o "modo" do dia: Correria / Normal / Tranquilo
- Baseado em: dia da semana, clima, feriados, histórico

NÍVEL TÁTICO (Equipe):
- Executa o modo escolhido:
  - Correria: cozinha rápida, atendimento ágil, promoções
  - Normal: ritmo equilibrado
  - Tranquilo: foco em qualidade, experiência premium

DIFERENÇA vs RL normal:
- RL normal: um agente decide TUDO (centenas de variáveis)
- Hierarchical: divide em níveis (mais simples e escalável!)

USO:
python examples/advanced/06_hierarchical_restaurant.py
"""

import business_rl as brl
import numpy as np


# ========== NÍVEL ALTO: GERENTE ==========

@brl.problem(name="Gerente")
class Gerente:
    """
    Agente de ALTO NÍVEL: escolhe o MODO de operação do dia

    Decide a cada manhã qual estratégia seguir.
    """

    obs = brl.Dict(
        # Previsão do dia
        dia_semana=brl.Discrete(7),
        eh_feriado=brl.Discrete(2),
        clima=brl.Discrete(3, labels=["Ruim", "OK", "Ótimo"]),

        # Histórico recente
        clientes_ontem=brl.Box(0, 300),
        receita_7d_media=brl.Box(0, 50000),
        avaliacao_media=brl.Box(0, 5),  # Estrelas

        # Recursos
        funcionarios_disponiveis=brl.Box(5, 20),
        estoque_nivel=brl.Box(0, 1),  # 0=vazio, 1=cheio

        # Contexto
        mes=brl.Discrete(12),
        fim_de_mes=brl.Discrete(2)
    )

    # Escolhe o MODO de operação
    action = brl.Discrete(3, labels=[
        "correria",   # Modo de alta demanda
        "normal",     # Modo equilibrado
        "tranquilo"   # Modo de baixa demanda
    ])

    objectives = brl.Terms(
        receita_total=0.50,
        satisfacao=0.30,
        eficiencia=0.20
    )

    def reward_receita_total(self, state, action, next_state):
        """Maximiza receita total."""
        receita = next_state['receita_7d_media']

        # Correria gera mais receita SE tiver recursos
        if action == 0:  # Correria
            if state['funcionarios_disponiveis'] >= 15:
                return receita / 100
            else:
                return -50  # Sem equipe suficiente!

        return receita / 150

    def reward_satisfacao(self, state, action, next_state):
        """Mantém satisfação alta."""
        avaliacao = next_state['avaliacao_media']

        # Tranquilo mantém qualidade alta
        if action == 2:  # Tranquilo
            return avaliacao * 30

        return avaliacao * 20

    def reward_eficiencia(self, state, action, next_state):
        """Evita desperdício de recursos."""
        # Correria em dia fraco = desperdício
        if action == 0:  # Correria
            if state['dia_semana'] in [0, 1]:  # Segunda/Terça
                return -30

        # Tranquilo em dia forte = oportunidade perdida
        if action == 2:  # Tranquilo
            if state['dia_semana'] in [4, 5]:  # Sexta/Sábado
                return -30

        return 10


# ========== NÍVEL BAIXO: COZINHA (modo correria) ==========

@brl.problem(name="CozinhaCorreria")
class CozinhaCorreria:
    """
    Agente de BAIXO NÍVEL: executa modo CORRERIA na cozinha

    Foca em: velocidade, throughput, eficiência
    """

    obs = brl.Dict(
        # Estado da cozinha
        pedidos_fila=brl.Box(0, 50),
        tempo_medio_preparo=brl.Box(5, 30),  # minutos

        # Recursos
        cozinheiros_ativos=brl.Box(2, 8),
        ingredientes_disponiveis=brl.Box(0, 1),

        # Performance
        pedidos_atrasados=brl.Box(0, 20),
        satisfacao_clientes=brl.Box(0, 5)
    )

    action = brl.Dict(
        velocidade_preparo=brl.Box(0.5, 2.0),  # Multiplicador de velocidade
        prioridade_velocidade_vs_qualidade=brl.Box(0, 1),  # 0=qualidade, 1=velocidade
        usar_receitas_rapidas=brl.Discrete(2)  # Simplificar pratos?
    )

    objectives = brl.Terms(
        throughput=0.60,     # Processar muitos pedidos
        satisfacao=0.25,     # Manter qualidade mínima
        custo=0.15           # Controlar desperdício
    )

    def reward_throughput(self, state, action, next_state):
        """Maximiza pedidos processados."""
        reducao_fila = state['pedidos_fila'] - next_state['pedidos_fila']
        return reducao_fila * 5

    def reward_satisfacao(self, state, action, next_state):
        """Mantém satisfação mínima."""
        satisfacao = next_state['satisfacao_clientes']

        # Penaliza se cair muito
        if satisfacao < 3.0:
            return -100

        return satisfacao * 10

    def reward_custo(self, state, action, next_state):
        """Controla desperdício."""
        # Receitas rápidas custam menos
        if action['usar_receitas_rapidas'] == 1:
            return 20
        return 0


# ========== NÍVEL BAIXO: ATENDIMENTO (modo tranquilo) ==========

@brl.problem(name="AtendimentoTranquilo")
class AtendimentoTranquilo:
    """
    Agente de BAIXO NÍVEL: executa modo TRANQUILO no atendimento

    Foca em: experiência premium, qualidade, fidelização
    """

    obs = brl.Dict(
        # Atendimento
        clientes_ativos=brl.Box(0, 50),
        tempo_medio_atendimento=brl.Box(2, 20),  # minutos

        # Experiência
        satisfacao_atual=brl.Box(0, 5),
        reclamacoes=brl.Box(0, 10),

        # Oportunidades
        clientes_novos=brl.Box(0, 20),  # Primeira visita
        ticket_medio=brl.Box(20, 200)
    )

    action = brl.Dict(
        atencao_por_cliente=brl.Box(0.5, 2.0),  # Tempo dedicado
        oferecer_extras=brl.Discrete(2),         # Sugerir sobremesas, vinhos?
        desconto_fidelidade=brl.Box(0, 0.20)     # 0% a 20%
    )

    objectives = brl.Terms(
        experiencia=0.50,
        fidelizacao=0.30,
        ticket_medio=0.20
    )

    def reward_experiencia(self, state, action, next_state):
        """Maximiza experiência do cliente."""
        satisfacao = next_state['satisfacao_atual']

        # Alta atenção melhora experiência
        atencao = action['atencao_por_cliente']
        bonus = (atencao - 1.0) * 20

        return satisfacao * 30 + bonus

    def reward_fidelizacao(self, state, action, next_state):
        """Conquista clientes novos."""
        # Oferecer extras para novos clientes
        if state['clientes_novos'] > 5 and action['oferecer_extras'] == 1:
            return 50

        return 0

    def reward_ticket_medio(self, state, action, next_state):
        """Aumenta valor por cliente."""
        ticket = next_state['ticket_medio']

        # Extras aumentam ticket
        if action['oferecer_extras'] == 1:
            return ticket / 5

        return ticket / 10


# ============================================================
# Sistema Hierárquico: Gerente + Equipes
# ============================================================

def treinar_sistema():
    """Treina toda a hierarquia."""
    print("="*70)
    print("TREINAMENTO: Sistema Hierárquico de Restaurante")
    print("="*70)

    # Nível Alto: Gerente
    print("\n👔 Treinando GERENTE (nível estratégico)...")
    gerente_problema = Gerente()
    gerente = brl.train(gerente_problema, hours=0.25)
    gerente.save('./modelos/gerente.pt')

    # Nível Baixo: Cozinha (correria)
    print("\n👨‍🍳 Treinando COZINHA CORRERIA (nível tático)...")
    cozinha_problema = CozinhaCorreria()
    cozinha = brl.train(cozinha_problema, hours=0.25)
    cozinha.save('./modelos/cozinha_correria.pt')

    # Nível Baixo: Atendimento (tranquilo)
    print("\n🤵 Treinando ATENDIMENTO TRANQUILO (nível tático)...")
    atendimento_problema = AtendimentoTranquilo()
    atendimento = brl.train(atendimento_problema, hours=0.25)
    atendimento.save('./modelos/atendimento_tranquilo.pt')

    print("\n✅ Hierarquia completa treinada!")

    return gerente, cozinha, atendimento


def simular_semana():
    """Simula uma semana de operação."""
    print("\n" + "="*70)
    print("SIMULAÇÃO: Uma Semana no Restaurante")
    print("="*70)

    # Carrega agentes
    gerente = brl.load('./modelos/gerente.pt')
    cozinha = brl.load('./modelos/cozinha_correria.pt')
    atendimento = brl.load('./modelos/atendimento_tranquilo.pt')

    dias = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"]

    print("\n📅 Simulando 7 dias de operação:\n")

    for dia_idx, dia_nome in enumerate(dias):
        print(f"{'='*70}")
        print(f"📅 {dia_nome.upper()}")
        print(f"{'='*70}")

        # 1. GERENTE decide o modo do dia
        estado_dia = {
            'dia_semana': dia_idx,
            'eh_feriado': 0,
            'clima': 1 if dia_idx < 5 else 2,  # Melhor no fim de semana
            'clientes_ontem': 80 + dia_idx * 20,
            'receita_7d_media': 15000 + dia_idx * 2000,
            'avaliacao_media': 4.2,
            'funcionarios_disponiveis': 12 + (3 if dia_idx >= 4 else 0),
            'estoque_nivel': 0.8,
            'mes': 3,
            'fim_de_mes': 0
        }

        decisao_gerente = gerente.decide(estado_dia, deterministic=True)
        modo = ["CORRERIA", "NORMAL", "TRANQUILO"][decisao_gerente.action]

        print(f"\n👔 GERENTE decidiu: Modo {modo}")
        print(f"   Baseado em:")
        print(f"   - Dia da semana: {dia_nome}")
        print(f"   - Funcionários disponíveis: {estado_dia['funcionarios_disponiveis']}")
        print(f"   - Clientes esperados: ~{estado_dia['clientes_ontem']}")

        # 2. EQUIPE executa o modo escolhido
        print(f"\n📋 Equipe executando modo {modo}:")

        if decisao_gerente.action == 0:  # CORRERIA
            estado_cozinha = {
                'pedidos_fila': 35,
                'tempo_medio_preparo': 15,
                'cozinheiros_ativos': 6,
                'ingredientes_disponiveis': 0.9,
                'pedidos_atrasados': 5,
                'satisfacao_clientes': 4.0
            }

            decisao = cozinha.decide(estado_cozinha, deterministic=True)

            print(f"   👨‍🍳 Cozinha:")
            print(f"      Velocidade: {decisao.action['velocidade_preparo']:.1f}x")
            print(f"      Prioridade: {'VELOCIDADE' if decisao.action['prioridade_velocidade_vs_qualidade'] > 0.5 else 'QUALIDADE'}")
            print(f"      Receitas rápidas: {'SIM' if decisao.action['usar_receitas_rapidas'] == 1 else 'NÃO'}")

        elif decisao_gerente.action == 2:  # TRANQUILO
            estado_atendimento = {
                'clientes_ativos': 20,
                'tempo_medio_atendimento': 12,
                'satisfacao_atual': 4.5,
                'reclamacoes': 1,
                'clientes_novos': 8,
                'ticket_medio': 85
            }

            decisao = atendimento.decide(estado_atendimento, deterministic=True)

            print(f"   🤵 Atendimento:")
            print(f"      Atenção por cliente: {decisao.action['atencao_por_cliente']:.1f}x")
            print(f"      Oferecer extras: {'SIM' if decisao.action['oferecer_extras'] == 1 else 'NÃO'}")
            print(f"      Desconto fidelidade: {decisao.action['desconto_fidelidade']:.1%}")

        else:  # NORMAL
            print(f"   ⚖️  Operação balanceada (mix de estratégias)")

        print()


def explicar_conceito():
    """Explica o conceito de Hierarchical RL."""
    print("="*70)
    print("POR QUE HIERARCHICAL RL?")
    print("="*70)

    print("""
🎯 PROBLEMA COM RL TRADICIONAL:

Imagine um único agente que precisa decidir TUDO:
- Modo do dia (estratégia)
- Velocidade da cozinha
- Estilo de atendimento
- Preços
- Promoções
- Etc...

Resultado:
❌ Problema GIGANTE (centenas de variáveis)
❌ Difícil de treinar
❌ Difícil de entender
❌ Difícil de manter


✅ SOLUÇÃO: HIERARCHICAL RL

Divide em NÍVEIS:

NÍVEL ALTO (Gerente):
- Decisões ESTRATÉGICAS
- Visão de longo prazo
- Poucos estados, poucos ações
- Fácil de treinar!

NÍVEL BAIXO (Equipes):
- Decisões TÁTICAS
- Execução específica
- Especializado por domínio
- Também fácil de treinar!

BENEFÍCIOS:
✅ Problemas menores = treino mais rápido
✅ Cada agente é especialista
✅ Mais fácil de entender e explicar
✅ Mais fácil de melhorar (troca só uma parte)
✅ Escalável (adiciona mais níveis se precisar)


🚀 COM ESTE FRAMEWORK:

Você só define:
1. Cada nível como um @brl.problem separado
2. Treina cada um independentemente
3. Usa em cascata (alto nível → baixo nível)

O framework cuida de toda a complexidade!
""")


if __name__ == "__main__":
    import os
    os.makedirs('./modelos', exist_ok=True)

    print("\n🏢 Hierarchical RL: Gerenciamento de Restaurante\n")

    # 1. Treina hierarquia
    treinar_sistema()

    # 2. Simula uma semana
    simular_semana()

    # 3. Explica conceito
    explicar_conceito()

    print("="*70)
    print("✅ RESUMO:")
    print("   - Hierarchical RL divide problema em níveis")
    print("   - Alto nível: decisões estratégicas")
    print("   - Baixo nível: execução tática")
    print("   - Muito mais simples que um agente único gigante!")
    print("="*70)
