"""
⏱️ Exemplo Intermediário 4: Otimização de Cronograma de Obra

PROBLEMA REAL:
Você tem uma obra com múltiplas atividades e precisa decidir:
- Qual atividade executar agora?
- Quantas equipes alocar?
- Vale a pena pagar hora extra?

Considerando:
- Prazo final do projeto
- Custos de mão de obra
- Dependências entre atividades
- Recursos limitados

DECISÃO: Qual atividade priorizar e com quantos recursos?

USO:
python examples/intermediate/04_cronograma_obra.py
"""

import business_rl as brl
import numpy as np


@brl.problem(name="CronogramaObra")
class CronogramaObra:
    """
    Problema: Sequenciamento e alocação de recursos em obra

    A cada dia, decide qual atividade executar e com quanto recurso
    """

    obs = brl.Dict(
        # Prazo do projeto
        dias_decorridos=brl.Box(0, 365),
        prazo_total_dias=brl.Box(90, 730),
        dias_restantes=brl.Box(0, 730),

        # Progresso geral
        percentual_concluido=brl.Box(0, 1),      # 0% a 100%
        atividades_concluidas=brl.Box(0, 50),
        atividades_pendentes=brl.Box(0, 50),

        # Atividade atual sendo considerada
        duracao_prevista_dias=brl.Box(1, 180),
        progresso_atividade=brl.Box(0, 1),       # 0% a 100%
        dias_executando=brl.Box(0, 180),

        # Prioridade e dependências
        prioridade=brl.Discrete(5, labels=[
            "Muito_Baixa", "Baixa", "Media", "Alta", "Critica"
        ]),
        tem_dependencias_pendentes=brl.Discrete(2),  # Há atividades anteriores não concluídas?
        bloqueia_outras=brl.Discrete(2),              # Outras atividades dependem desta?

        # Recursos
        equipes_disponiveis=brl.Box(1, 20),
        equipes_alocadas_atividade=brl.Box(0, 20),
        produtividade_media=brl.Box(0.5, 1.5),  # Eficiência da equipe

        # Custos
        custo_dia_equipe_normal=brl.Box(500, 3000),
        custo_dia_equipe_extra=brl.Box(750, 4500),  # Hora extra
        orcamento_disponivel=brl.Box(0, 10000000),
        orcamento_usado_percentual=brl.Box(0, 1),

        # Clima e condições
        clima_favoravel=brl.Discrete(2),        # 0=desfavorável, 1=favorável
        tipo_atividade=brl.Discrete(4, labels=[
            "Fundacao", "Estrutura", "Vedacao", "Acabamento"
        ]),

        # Multas e bônus
        multa_atraso_dia=brl.Box(0, 50000),
        bonus_antecipacao_dia=brl.Box(0, 30000)
    )

    # Decisão: Quantas equipes alocar nesta atividade?
    action = brl.Discrete(6, labels=[
        "Nenhuma",          # 0 equipes (não executar)
        "Minima",           # 1 equipe
        "Normal",           # 2-3 equipes
        "Acelerada",        # 4-5 equipes
        "Maxima",           # 6+ equipes
        "Extra"             # Equipes + hora extra
    ])

    objectives = brl.Terms(
        prazo=0.45,          # Cumprir prazo
        custo=0.35,          # Minimizar custo
        qualidade=0.20       # Manter qualidade
    )

    def reward_prazo(self, state, action, next_state):
        """Penaliza atrasos e recompensa antecipação."""
        dias_restantes = state['dias_restantes']
        prazo_total = state['prazo_total_dias']
        percentual_concluido = state['percentual_concluido']

        # Progresso esperado vs real
        percentual_tempo_usado = 1 - (dias_restantes / prazo_total)
        atraso_percentual = percentual_tempo_usado - percentual_concluido

        # Atividade crítica ou bloqueia outras?
        critica = (state['prioridade'] >= 3) or (state['bloqueia_outras'] == 1)

        # Não fazer nada em atividade crítica = muito ruim
        if action == 0 and critica:
            return -100

        # Fazer hora extra quando tem folga = desperdício
        if action == 5 and atraso_percentual < -0.1:  # 10% adiantado
            return -50

        # Alocação baseada na urgência
        if atraso_percentual > 0.2:  # 20% atrasado
            if action >= 4:  # Máxima ou Extra
                return 80  # Bom! Tentando recuperar
            elif action <= 1:
                return -80  # Péssimo! Parado no atraso
        elif atraso_percentual > 0:  # Levemente atrasado
            if action >= 3:  # Acelerada
                return 50
            elif action == 0:
                return -30
        else:  # No prazo ou adiantado
            if action == 2:  # Normal
                return 30  # Ideal
            elif action >= 4:
                return -20  # Desnecessário

        return 0

    def reward_custo(self, state, action, next_state):
        """Minimiza custos."""
        # Custos por nível de alocação
        custos_acao = {
            0: 0,  # Nenhuma equipe
            1: state['custo_dia_equipe_normal'] * 1,      # 1 equipe
            2: state['custo_dia_equipe_normal'] * 2.5,    # 2-3 equipes
            3: state['custo_dia_equipe_normal'] * 4.5,    # 4-5 equipes
            4: state['custo_dia_equipe_normal'] * 7,      # 6+ equipes
            5: state['custo_dia_equipe_extra'] * 5        # Hora extra
        }

        custo_diario = custos_acao[action]

        # Penaliza custo (normaliza)
        penalidade_custo = -custo_diario / 1000

        # Orçamento estourado?
        orcamento_usado = state['orcamento_usado_percentual']
        if orcamento_usado > 0.95:  # >95% do orçamento usado
            if action >= 4:  # Evita gastos altos
                return -100
        elif orcamento_usado > 0.85:
            if action >= 4:
                return -50

        # Hora extra só se realmente necessário
        if action == 5:  # Extra
            if state['prioridade'] < 3:  # Não é prioritária
                return -80  # Desperdício

        return penalidade_custo

    def reward_qualidade(self, state, action, next_state):
        """Mantém qualidade da execução."""
        produtividade = state['produtividade_media']
        clima = state['clima_favoravel']

        # Clima desfavorável + muitas equipes = qualidade baixa
        if clima == 0:  # Clima ruim
            if action >= 3:  # Muitas equipes
                return -40  # Difícil manter qualidade
            elif action == 0:
                return 20  # Bom! Aguardando clima melhorar
            else:
                return 10  # Trabalho cauteloso

        # Produtividade baixa + acelerar = problemas
        if produtividade < 0.7:
            if action >= 4:  # Máxima ou Extra
                return -50  # Equipe sobrecarregada
            elif action == 2:  # Normal
                return 20  # Ritmo adequado

        # Tem dependências pendentes? Não deveria executar
        if state['tem_dependencias_pendentes'] == 1:
            if action > 0:
                return -60  # Executando fora de ordem!
            else:
                return 10  # Correto esperar

        return 0


def treinar_modelo():
    """Treina o modelo de cronograma."""
    print("="*70)
    print("TREINAMENTO: Otimização de Cronograma de Obra")
    print("="*70)

    problema = CronogramaObra()

    print("\n📊 Informações do problema:")
    print(f"  Observações: {problema.get_info()['observation_dim']} variáveis")
    print(f"  Decisões: {problema.get_info()['action_space']['action']['n']} níveis de alocação")
    print(f"  Objetivos: Prazo, Custo, Qualidade")

    print("\n🏋️ Treinando modelo (1 hora)...")
    modelo = brl.train(
        problema,
        algorithm='PPO',
        hours=1,
        config={'learning_rate': 3e-4}
    )

    modelo.save('./modelos/cronograma_obra.pt')
    print("\n✅ Modelo salvo!")

    return modelo


def simular_obra():
    """Simula gestão de obra."""
    print("\n" + "="*70)
    print("SIMULAÇÃO: Gestão de Cronograma")
    print("="*70)

    modelo = brl.load('./modelos/cronograma_obra.pt')

    # Atividades da obra
    atividades = [
        {
            'nome': 'Fundação - Escavação',
            'estado': {
                'dias_decorridos': 10,
                'prazo_total_dias': 180,
                'dias_restantes': 170,
                'percentual_concluido': 0.05,
                'atividades_concluidas': 1,
                'atividades_pendentes': 15,
                'duracao_prevista_dias': 15,
                'progresso_atividade': 0.30,
                'dias_executando': 5,
                'prioridade': 4,  # Crítica
                'tem_dependencias_pendentes': 0,
                'bloqueia_outras': 1,
                'equipes_disponiveis': 8,
                'equipes_alocadas_atividade': 3,
                'produtividade_media': 0.9,
                'custo_dia_equipe_normal': 1200,
                'custo_dia_equipe_extra': 1800,
                'orcamento_disponivel': 2500000,
                'orcamento_usado_percentual': 0.08,
                'clima_favoravel': 1,
                'tipo_atividade': 0,  # Fundação
                'multa_atraso_dia': 5000,
                'bonus_antecipacao_dia': 2000
            }
        },
        {
            'nome': 'Estrutura - Pilares Térreo',
            'estado': {
                'dias_decorridos': 10,
                'prazo_total_dias': 180,
                'dias_restantes': 170,
                'percentual_concluido': 0.05,
                'atividades_concluidas': 1,
                'atividades_pendentes': 15,
                'duracao_prevista_dias': 10,
                'progresso_atividade': 0,
                'dias_executando': 0,
                'prioridade': 4,  # Crítica
                'tem_dependencias_pendentes': 1,  # Depende da fundação!
                'bloqueia_outras': 1,
                'equipes_disponiveis': 5,
                'equipes_alocadas_atividade': 0,
                'produtividade_media': 1.0,
                'custo_dia_equipe_normal': 1500,
                'custo_dia_equipe_extra': 2250,
                'orcamento_disponivel': 2500000,
                'orcamento_usado_percentual': 0.08,
                'clima_favoravel': 1,
                'tipo_atividade': 1,  # Estrutura
                'multa_atraso_dia': 5000,
                'bonus_antecipacao_dia': 2000
            }
        },
        {
            'nome': 'Acabamento - Pintura Externa',
            'estado': {
                'dias_decorridos': 10,
                'prazo_total_dias': 180,
                'dias_restantes': 170,
                'percentual_concluido': 0.05,
                'atividades_concluidas': 1,
                'atividades_pendentes': 15,
                'duracao_prevista_dias': 20,
                'progresso_atividade': 0,
                'dias_executando': 0,
                'prioridade': 1,  # Baixa (pode fazer depois)
                'tem_dependencias_pendentes': 1,
                'bloqueia_outras': 0,
                'equipes_disponiveis': 10,
                'equipes_alocadas_atividade': 0,
                'produtividade_media': 1.1,
                'custo_dia_equipe_normal': 800,
                'custo_dia_equipe_extra': 1200,
                'orcamento_disponivel': 2500000,
                'orcamento_usado_percentual': 0.08,
                'clima_favoravel': 0,  # Clima ruim!
                'tipo_atividade': 3,  # Acabamento
                'multa_atraso_dia': 5000,
                'bonus_antecipacao_dia': 2000
            }
        }
    ]

    acoes_labels = ["Nenhuma", "Mínima (1)", "Normal (2-3)", "Acelerada (4-5)", "Máxima (6+)", "Extra (hora extra)"]

    print("\n📋 Obra: Prédio residencial - 3 pavimentos")
    print(f"   Prazo: 180 dias | Orçamento: R$ 2.5M\n")

    for atividade in atividades:
        decisao = modelo.decide(atividade['estado'], deterministic=True)
        acao = decisao.action

        print(f"{'='*70}")
        print(f"🏗️ {atividade['nome']}")
        print(f"{'='*70}")

        estado = atividade['estado']

        print(f"\n📊 Status:")
        print(f"  Progresso: {estado['progresso_atividade']:.0%}")
        print(f"  Duração prevista: {estado['duracao_prevista_dias']} dias")
        print(f"  Prioridade: {['Muito Baixa', 'Baixa', 'Média', 'Alta', 'Crítica'][estado['prioridade']]}")

        if estado['tem_dependencias_pendentes'] == 1:
            print(f"  ⚠️  Aguardando atividades anteriores")

        if estado['bloqueia_outras'] == 1:
            print(f"  ⚡ CRÍTICO: Bloqueia outras atividades!")

        if estado['clima_favoravel'] == 0:
            print(f"  ☔ Clima desfavorável")

        print(f"\n💰 Recursos:")
        print(f"  Equipes disponíveis: {estado['equipes_disponiveis']:.0f}")
        print(f"  Custo equipe/dia: R$ {estado['custo_dia_equipe_normal']:,.0f}")

        print(f"\n✅ DECISÃO: {acoes_labels[acao]}")

        # Justificativa
        if acao == 0:
            if estado['tem_dependencias_pendentes'] == 1:
                justificativa = "Aguardando dependências"
            elif estado['clima_favoravel'] == 0 and estado['tipo_atividade'] == 3:
                justificativa = "Clima desfavorável para acabamento"
            else:
                justificativa = "Não prioritário no momento"
        elif acao >= 4:
            justificativa = "Atividade crítica - máxima prioridade"
        elif acao == 2:
            justificativa = "Alocação normal - ritmo adequado"
        else:
            justificativa = "Manutenção do cronograma"

        print(f"   💡 {justificativa}")
        print()


if __name__ == "__main__":
    import os
    os.makedirs('./modelos', exist_ok=True)

    print("\n⏱️ Otimização Inteligente de Cronograma de Obra\n")

    # 1. Treina
    treinar_modelo()

    # 2. Simula
    simular_obra()

    print("="*70)
    print("✅ PRONTO!")
    print("   Modelo otimiza cronograma considerando:")
    print("   - Cumprimento de prazos")
    print("   - Minimização de custos")
    print("   - Manutenção de qualidade")
    print("   - Dependências entre atividades")
    print("="*70)
