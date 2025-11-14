"""
Exemplo Avançado 5: Multi-Agent RL - Supply Chain

Este exemplo demonstra:
- Múltiplos agentes cooperando (MARL - Multi-Agent RL)
- Comunicação entre agentes
- Políticas centralizadas vs descentralizadas
- Emergência de comportamento cooperativo
- Credit assignment em sistemas multi-agente

Cenário:
- 3 agentes na cadeia de suprimentos:
  * Fornecedor (upstream)
  * Distribuidor (middle)
  * Varejista (downstream)
- Objetivo: Minimizar custos globais e rupturas
"""

import business_rl as brl
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class AgentMessage:
    """Mensagem trocada entre agentes."""
    sender: str
    receiver: str
    content: Dict
    timestamp: int


# ===== AGENTE 1: FORNECEDOR =====
@brl.problem(name="FornecedorAgent")
class FornecedorAgent:
    """
    Agente Fornecedor: Produz e envia produtos ao Distribuidor
    """

    obs = brl.Dict(
        # Estado próprio
        estoque_proprio=brl.Box(0, 1000),
        capacidade_producao=brl.Box(0, 500),
        custo_producao=brl.Box(10, 50),

        # Pedidos recebidos
        pedido_distribuidor=brl.Box(0, 500),
        pedidos_backlog=brl.Box(0, 300),

        # Informação compartilhada (comunicação)
        previsao_demanda_downstream=brl.Box(0, 500),
        nivel_estoque_distribuidor=brl.Box(0, 1000),

        # Contexto temporal
        dia=brl.Discrete(30),
        mes=brl.Discrete(12),
        tendencia_mercado=brl.Box(-1, 1),  # -1=queda, +1=alta

        # KPIs históricos
        taxa_atendimento_7d=brl.Box(0, 1),
        custo_medio_7d=brl.Box(0, 100)
    )

    action = brl.Dict(
        # Quanto produzir
        quantidade_producao=brl.Box(0, 500),

        # Quanto enviar ao distribuidor
        quantidade_envio=brl.Box(0, 500),

        # Mensagem ao distribuidor
        disponibilidade_informada=brl.Box(0, 1000),
        prazo_entrega_dias=brl.Discrete(15, offset=1)  # 1-15 dias
    )

    objectives = brl.Terms(
        custo_producao=0.40,
        nivel_servico=0.35,
        custo_estoque=0.15,
        coordenacao=0.10
    )

    def reward_custo_producao(self, state, action, next_state):
        """Minimiza custos de produção."""
        custo = action['quantidade_producao'] * state['custo_producao']
        return -custo / 100

    def reward_nivel_servico(self, state, action, next_state):
        """Maximiza atendimento dos pedidos."""
        pedido = state['pedido_distribuidor']
        atendido = min(action['quantidade_envio'], pedido)

        if pedido > 0:
            taxa_atendimento = atendido / pedido
            return taxa_atendimento * 100
        return 0

    def reward_custo_estoque(self, state, action, next_state):
        """Penaliza excesso de estoque."""
        estoque = next_state['estoque_proprio']
        if estoque > 500:
            return -(estoque - 500) / 10
        return 0

    def reward_coordenacao(self, state, action, next_state):
        """Recompensa boa coordenação com downstream."""
        # Produção alinhada com demanda prevista
        demanda_prevista = state['previsao_demanda_downstream']
        producao = action['quantidade_producao']

        erro = abs(producao - demanda_prevista)
        return max(0, 100 - erro)


# ===== AGENTE 2: DISTRIBUIDOR =====
@brl.problem(name="DistribuidorAgent")
class DistribuidorAgent:
    """
    Agente Distribuidor: Recebe do Fornecedor, envia ao Varejista
    """

    obs = brl.Dict(
        # Estado próprio
        estoque_proprio=brl.Box(0, 1000),
        em_transito_fornecedor=brl.Box(0, 500),

        # Pedidos
        pedido_varejista=brl.Box(0, 500),
        pedidos_backlog=brl.Box(0, 300),

        # Comunicação upstream
        disponibilidade_fornecedor=brl.Box(0, 1000),
        prazo_fornecedor=brl.Box(1, 15),

        # Comunicação downstream
        urgencia_varejista=brl.Box(0, 1),  # 0=normal, 1=urgente
        nivel_estoque_varejista=brl.Box(0, 500),

        # Contexto
        dia=brl.Discrete(30),
        custo_transporte=brl.Box(5, 30),
        taxa_atendimento_7d=brl.Box(0, 1)
    )

    action = brl.Dict(
        # Pedido ao fornecedor
        quantidade_pedido_upstream=brl.Box(0, 500),

        # Envio ao varejista
        quantidade_envio_downstream=brl.Box(0, 500),

        # Modo de transporte (0=normal, 1=expresso)
        modo_transporte=brl.Discrete(2, labels=["normal", "expresso"]),

        # Mensagens
        previsao_demanda=brl.Box(0, 500),
        alerta_ruptura=brl.Discrete(2, labels=["nao", "sim"])
    )

    objectives = brl.Terms(
        nivel_servico=0.40,
        custo_transporte=0.25,
        custo_estoque=0.20,
        coordenacao_upstream=0.15
    )

    def reward_nivel_servico(self, state, action, next_state):
        """Atende pedidos do varejista."""
        pedido = state['pedido_varejista']
        atendido = min(action['quantidade_envio_downstream'], pedido)

        if pedido > 0:
            taxa = atendido / pedido

            # Bônus se atendeu em modo urgente
            if state['urgencia_varejista'] > 0.5 and taxa > 0.9:
                return taxa * 120
            return taxa * 100
        return 0

    def reward_custo_transporte(self, state, action, next_state):
        """Minimiza custos de transporte."""
        quantidade = action['quantidade_envio_downstream']
        custo_base = state['custo_transporte']

        # Expresso custa 50% mais
        if action['modo_transporte'] == 1:
            custo = quantidade * custo_base * 1.5
        else:
            custo = quantidade * custo_base

        return -custo / 100

    def reward_custo_estoque(self, state, action, next_state):
        """Penaliza excesso de estoque."""
        estoque = next_state['estoque_proprio']
        if estoque > 600:
            return -(estoque - 600) / 10
        return 0

    def reward_coordenacao_upstream(self, state, action, next_state):
        """Coordenação com fornecedor."""
        # Pedido alinhado com disponibilidade
        disponivel = state['disponibilidade_fornecedor']
        pedido = action['quantidade_pedido_upstream']

        if pedido <= disponivel:
            return 20
        else:
            return -10  # Pediu mais que disponível


# ===== AGENTE 3: VAREJISTA =====
@brl.problem(name="VarejistaAgent")
class VarejistaAgent:
    """
    Agente Varejista: Atende demanda final dos clientes
    """

    obs = brl.Dict(
        # Estado próprio
        estoque_proprio=brl.Box(0, 500),
        em_transito_distribuidor=brl.Box(0, 300),

        # Demanda do cliente
        demanda_atual=brl.Box(0, 200),
        demanda_prevista_7d=brl.Box(0, 200, shape=(7,)),

        # Comunicação upstream
        disponibilidade_distribuidor=brl.Box(0, 1000),
        tem_alerta_ruptura=brl.Discrete(2),

        # Contexto de mercado
        dia=brl.Discrete(30),
        dia_semana=brl.Discrete(7),
        promocao_ativa=brl.Discrete(2),
        preco_concorrente=brl.Box(50, 200),

        # KPIs
        taxa_ruptura_7d=brl.Box(0, 1),
        satisfacao_cliente_7d=brl.Box(0, 1)
    )

    action = brl.Dict(
        # Pedido ao distribuidor
        quantidade_pedido=brl.Box(0, 300),

        # Sinalizar urgência
        urgencia=brl.Box(0, 1),

        # Ajuste de preço (±20%)
        ajuste_preco=brl.Box(-0.2, 0.2),

        # Mensagens
        previsao_vendas=brl.Box(0, 200)
    )

    objectives = brl.Terms(
        vendas=0.35,
        satisfacao_cliente=0.30,
        custo_ruptura=0.20,
        custo_estoque=0.15
    )

    def reward_vendas(self, state, action, next_state):
        """Maximiza vendas."""
        demanda = state['demanda_atual']
        estoque = state['estoque_proprio']

        vendas = min(demanda, estoque)

        # Ajuste de preço afeta demanda
        if action['ajuste_preco'] < 0:  # Desconto
            vendas *= (1 - action['ajuste_preco'])  # Aumenta vendas

        return vendas * 10

    def reward_satisfacao_cliente(self, state, action, next_state):
        """Satisfação do cliente."""
        demanda = state['demanda_atual']
        estoque = state['estoque_proprio']

        if demanda > 0:
            taxa_atendimento = min(estoque / demanda, 1.0)
            return taxa_atendimento * 100
        return 100

    def reward_custo_ruptura(self, state, action, next_state):
        """Penaliza rupturas."""
        demanda = state['demanda_atual']
        estoque = state['estoque_proprio']

        if estoque < demanda:
            ruptura = demanda - estoque
            return -ruptura * 5
        return 0

    def reward_custo_estoque(self, state, action, next_state):
        """Penaliza excesso."""
        estoque = next_state['estoque_proprio']
        if estoque > 300:
            return -(estoque - 300) / 10
        return 0


class MultiAgentSupplyChain:
    """Coordenador da cadeia de suprimentos multi-agente."""

    def __init__(self):
        # Agentes individuais
        self.fornecedor = None
        self.distribuidor = None
        self.varejista = None

        # Estado global
        self.dia_atual = 0
        self.mes_atual = 1

        # Canal de comunicação
        self.message_queue: List[AgentMessage] = []

        # Métricas globais
        self.metricas = {
            'custo_total': [],
            'nivel_servico_global': [],
            'rupturas_totais': [],
            'estoque_total': []
        }

    def treinar_agentes(self, hours_per_agent: float = 1.0):
        """Treina cada agente individualmente."""
        print("=" * 70)
        print("TREINAMENTO: MULTI-AGENT SUPPLY CHAIN")
        print("=" * 70)

        # Treina Fornecedor
        print("\n🏭 Treinando Fornecedor...")
        problema_fornecedor = FornecedorAgent()
        self.fornecedor = brl.train(
            problema_fornecedor,
            algorithm='PPO',
            hours=hours_per_agent,
            config={'learning_rate': 3e-4, 'gamma': 0.99}
        )
        self.fornecedor.save('./modelos/fornecedor_agent.pt')

        # Treina Distribuidor
        print("\n🚚 Treinando Distribuidor...")
        problema_distribuidor = DistribuidorAgent()
        self.distribuidor = brl.train(
            problema_distribuidor,
            algorithm='PPO',
            hours=hours_per_agent,
            config={'learning_rate': 3e-4, 'gamma': 0.99}
        )
        self.distribuidor.save('./modelos/distribuidor_agent.pt')

        # Treina Varejista
        print("\n🏪 Treinando Varejista...")
        problema_varejista = VarejistaAgent()
        self.varejista = brl.train(
            problema_varejista,
            algorithm='PPO',
            hours=hours_per_agent,
            config={'learning_rate': 3e-4, 'gamma': 0.99}
        )
        self.varejista.save('./modelos/varejista_agent.pt')

        print("\n✅ Todos os agentes treinados!")

    def carregar_agentes(self):
        """Carrega agentes treinados."""
        self.fornecedor = brl.load('./modelos/fornecedor_agent.pt')
        self.distribuidor = brl.load('./modelos/distribuidor_agent.pt')
        self.varejista = brl.load('./modelos/varejista_agent.pt')

    def simular_episodio(self, n_dias: int = 30):
        """Simula um episódio de n dias."""
        # Estados iniciais
        estado_fornecedor = self._estado_inicial_fornecedor()
        estado_distribuidor = self._estado_inicial_distribuidor()
        estado_varejista = self._estado_inicial_varejista()

        historico = []

        for dia in range(n_dias):
            self.dia_atual = dia

            # 1. VAREJISTA decide (downstream)
            decisao_varejista = self.varejista.decide(
                estado_varejista, deterministic=True
            )

            # 2. DISTRIBUIDOR decide (middle)
            # Recebe informação do varejista
            estado_distribuidor['pedido_varejista'] = \
                decisao_varejista.action['quantidade_pedido']
            estado_distribuidor['urgencia_varejista'] = \
                decisao_varejista.action['urgencia']

            decisao_distribuidor = self.distribuidor.decide(
                estado_distribuidor, deterministic=True
            )

            # 3. FORNECEDOR decide (upstream)
            # Recebe informação do distribuidor
            estado_fornecedor['pedido_distribuidor'] = \
                decisao_distribuidor.action['quantidade_pedido_upstream']
            estado_fornecedor['previsao_demanda_downstream'] = \
                decisao_distribuidor.action['previsao_demanda']

            decisao_fornecedor = self.fornecedor.decide(
                estado_fornecedor, deterministic=True
            )

            # 4. Atualiza estados (simulação simplificada)
            estado_fornecedor = self._atualizar_fornecedor(
                estado_fornecedor, decisao_fornecedor, decisao_distribuidor
            )
            estado_distribuidor = self._atualizar_distribuidor(
                estado_distribuidor, decisao_distribuidor,
                decisao_fornecedor, decisao_varejista
            )
            estado_varejista = self._atualizar_varejista(
                estado_varejista, decisao_varejista, decisao_distribuidor
            )

            # Registra métricas
            historico.append({
                'dia': dia,
                'estoque_fornecedor': estado_fornecedor['estoque_proprio'],
                'estoque_distribuidor': estado_distribuidor['estoque_proprio'],
                'estoque_varejista': estado_varejista['estoque_proprio'],
                'demanda': estado_varejista['demanda_atual'],
                'vendas': min(estado_varejista['estoque_proprio'],
                            estado_varejista['demanda_atual'])
            })

        return historico

    def _estado_inicial_fornecedor(self):
        return {
            'estoque_proprio': 300,
            'capacidade_producao': 200,
            'custo_producao': 25,
            'pedido_distribuidor': 100,
            'pedidos_backlog': 0,
            'previsao_demanda_downstream': 100,
            'nivel_estoque_distribuidor': 200,
            'dia': 1,
            'mes': 1,
            'tendencia_mercado': 0.1,
            'taxa_atendimento_7d': 0.95,
            'custo_medio_7d': 50
        }

    def _estado_inicial_distribuidor(self):
        return {
            'estoque_proprio': 200,
            'em_transito_fornecedor': 50,
            'pedido_varejista': 80,
            'pedidos_backlog': 0,
            'disponibilidade_fornecedor': 300,
            'prazo_fornecedor': 3,
            'urgencia_varejista': 0.0,
            'nivel_estoque_varejista': 100,
            'dia': 1,
            'custo_transporte': 15,
            'taxa_atendimento_7d': 0.9
        }

    def _estado_inicial_varejista(self):
        return {
            'estoque_proprio': 100,
            'em_transito_distribuidor': 30,
            'demanda_atual': 50,
            'demanda_prevista_7d': np.array([50, 55, 52, 48, 60, 45, 50]),
            'disponibilidade_distribuidor': 200,
            'tem_alerta_ruptura': 0,
            'dia': 1,
            'dia_semana': 0,
            'promocao_ativa': 0,
            'preco_concorrente': 100,
            'taxa_ruptura_7d': 0.05,
            'satisfacao_cliente_7d': 0.92
        }

    def _atualizar_fornecedor(self, estado, decisao, decisao_dist):
        # Simulação simplificada
        estado = estado.copy()
        estado['estoque_proprio'] = max(0,
            estado['estoque_proprio'] +
            decisao.action['quantidade_producao'] -
            decisao.action['quantidade_envio']
        )
        estado['pedido_distribuidor'] = decisao_dist.action['quantidade_pedido_upstream']
        return estado

    def _atualizar_distribuidor(self, estado, decisao, decisao_forn, decisao_var):
        estado = estado.copy()
        # Recebe do fornecedor
        recebido = decisao_forn.action['quantidade_envio']
        # Envia ao varejista
        enviado = decisao.action['quantidade_envio_downstream']

        estado['estoque_proprio'] = max(0,
            estado['estoque_proprio'] + recebido - enviado
        )
        estado['pedido_varejista'] = decisao_var.action['quantidade_pedido']
        return estado

    def _atualizar_varejista(self, estado, decisao, decisao_dist):
        estado = estado.copy()
        # Recebe do distribuidor
        recebido = decisao_dist.action['quantidade_envio_downstream']
        # Vende
        vendas = min(estado['estoque_proprio'], estado['demanda_atual'])

        estado['estoque_proprio'] = max(0,
            estado['estoque_proprio'] + recebido - vendas
        )
        # Nova demanda aleatória
        estado['demanda_atual'] = max(0, np.random.normal(50, 15))
        return estado


def demo_multi_agent():
    """Demonstração do sistema multi-agente."""
    print("\n" + "=" * 70)
    print("DEMONSTRAÇÃO: SUPPLY CHAIN MULTI-AGENTE")
    print("=" * 70)

    # Cria coordenador
    supply_chain = MultiAgentSupplyChain()

    # Carrega agentes (assumindo já treinados)
    print("\n📦 Carregando agentes...")
    supply_chain.carregar_agentes()

    # Simula 30 dias
    print("\n🔄 Simulando 30 dias de operação...")
    historico = supply_chain.simular_episodio(n_dias=30)

    # Análise
    print("\n📊 Análise de Performance:")
    print("-" * 70)

    estoque_total_medio = np.mean([
        h['estoque_fornecedor'] +
        h['estoque_distribuidor'] +
        h['estoque_varejista']
        for h in historico
    ])

    taxa_atendimento = np.mean([
        h['vendas'] / h['demanda'] if h['demanda'] > 0 else 1.0
        for h in historico
    ])

    print(f"Estoque total médio: {estoque_total_medio:.0f} unidades")
    print(f"Taxa de atendimento: {taxa_atendimento:.2%}")

    # Efeito Bullwhip (variabilidade aumenta upstream)
    var_demanda = np.var([h['demanda'] for h in historico])
    var_estoque_fornecedor = np.var([h['estoque_fornecedor'] for h in historico])

    bullwhip_ratio = var_estoque_fornecedor / (var_demanda + 1e-6)
    print(f"Bullwhip Ratio: {bullwhip_ratio:.2f} (menor é melhor)")


if __name__ == "__main__":
    import os
    os.makedirs('./modelos', exist_ok=True)

    print("🚀 Iniciando exemplo Multi-Agent Supply Chain\n")

    # Cria sistema
    sistema = MultiAgentSupplyChain()

    # Treina agentes (usar hours menores para teste rápido)
    sistema.treinar_agentes(hours_per_agent=0.5)  # 30min cada

    # Demonstração
    demo_multi_agent()

    print("\n" + "=" * 70)
    print("✅ Exemplo concluído!")
    print("=" * 70)
