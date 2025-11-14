"""
Exemplo Avançado 6: Hierarchical RL - Sistema de Trading

Este exemplo demonstra:
- Hierarchical Reinforcement Learning (HRL)
- Agente de alto nível (meta-controller): estratégia geral
- Agentes de baixo nível (controllers): execução tática
- Temporal abstractions (opções/skills)
- Decomposição hierárquica de decisões complexas

Hierarquia:
┌─────────────────────────────────────┐
│   META-CONTROLLER (Alto Nível)      │
│   Decide: ESTRATÉGIA do dia         │
│   - Agressiva (compra forte)        │
│   - Moderada (balanço)              │
│   - Conservadora (proteção)         │
│   - Liquidação (vender tudo)        │
└─────────────────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│   CONTROLLERS (Baixo Nível)         │
│   Executam: AÇÕES específicas       │
│   - Quanto comprar/vender           │
│   - Que ativos                      │
│   - Stop loss / take profit         │
└─────────────────────────────────────┘
"""

import business_rl as brl
import numpy as np
from typing import Dict, List, Optional
from enum import Enum


class TradingStrategy(Enum):
    """Estratégias de alto nível."""
    AGRESSIVA = 0      # Alta exposição, busca ganhos
    MODERADA = 1       # Balanço risco/retorno
    CONSERVADORA = 2   # Proteção de capital
    LIQUIDACAO = 3     # Reduz posições


# ===== META-CONTROLLER (ALTO NÍVEL) =====
@brl.problem(name="MetaController")
class MetaController:
    """
    Agente de alto nível: escolhe a ESTRATÉGIA geral para o período.

    Decide a cada N horas qual estratégia seguir,
    baseado em condições macro do mercado.
    """

    obs = brl.Dict(
        # Condições macro do mercado
        volatilidade_mercado=brl.Box(0, 1),  # VIX normalizado
        tendencia_mercado=brl.Box(-1, 1),    # -1=baixa, +1=alta
        volume_mercado=brl.Box(0, 1),        # Volume normalizado

        # Retornos médios dos índices (agregados)
        sp500_retorno_medio=brl.Box(-0.1, 0.1),
        nasdaq_retorno_medio=brl.Box(-0.1, 0.1),

        # Estado do portfólio
        valor_portfolio=brl.Box(0, 1000000),
        exposicao_atual=brl.Box(0, 1),  # % capital investido
        posicoes_abertas=brl.Box(0, 20),

        # Performance recente
        retorno_7d=brl.Box(-0.5, 0.5),
        retorno_30d=brl.Box(-0.5, 0.5),
        sharpe_ratio=brl.Box(-3, 3),
        max_drawdown=brl.Box(0, 1),

        # Sentimento de mercado
        fear_greed_index=brl.Box(0, 100),
        put_call_ratio=brl.Box(0, 2),

        # Calendário
        hora_dia=brl.Discrete(24),
        dia_semana=brl.Discrete(7),
        fim_trimestre=brl.Discrete(2),  # Próximo ao fim?

        # Restrições
        margem_disponivel=brl.Box(0, 1),
        limite_risco_usado=brl.Box(0, 1)  # % do limite usado
    )

    # Escolhe uma das 4 estratégias
    action = brl.Discrete(
        4,
        labels=["agressiva", "moderada", "conservadora", "liquidacao"]
    )

    objectives = brl.Terms(
        retorno_ajustado=0.50,    # Retorno ajustado por risco
        consistencia=0.25,         # Volatilidade dos retornos
        protecao_capital=0.15,     # Evitar grandes perdas
        adaptabilidade=0.10        # Resposta a mudanças
    )

    def reward_retorno_ajustado(self, state, action, next_state):
        """Retorno ajustado pelo risco (Sharpe-like)."""
        retorno = next_state['retorno_7d']
        sharpe = next_state['sharpe_ratio']

        # Estratégias diferentes têm diferentes perfis
        if action == TradingStrategy.AGRESSIVA.value:
            # Recompensa retornos positivos, tolera volatilidade
            return retorno * 200 + sharpe * 10

        elif action == TradingStrategy.MODERADA.value:
            # Balanço
            return retorno * 150 + sharpe * 20

        elif action == TradingStrategy.CONSERVADORA.value:
            # Prioriza Sharpe sobre retorno absoluto
            return retorno * 100 + sharpe * 30

        else:  # LIQUIDACAO
            # Recompensa redução de exposição
            reducao_exposicao = (state['exposicao_atual'] -
                                next_state['exposicao_atual'])
            return reducao_exposicao * 100

    def reward_consistencia(self, state, action, next_state):
        """Penaliza volatilidade excessiva."""
        volatilidade = state['volatilidade_mercado']

        # Em alta volatilidade, ser conservador é bom
        if volatilidade > 0.7:
            if action == TradingStrategy.CONSERVADORA.value:
                return 30
            elif action == TradingStrategy.AGRESSIVA.value:
                return -20

        return 0

    def reward_protecao_capital(self, state, action, next_state):
        """Evita grandes perdas."""
        drawdown = next_state['max_drawdown']

        if drawdown > 0.2:  # >20% de perda
            return -100
        elif drawdown > 0.1:  # >10% de perda
            return -30
        else:
            return 10

    def reward_adaptabilidade(self, state, action, next_state):
        """Recompensa adaptar-se às condições."""
        tendencia = state['tendencia_mercado']
        volatilidade = state['volatilidade_mercado']

        # Em alta tendência + baixa vol: ser agressivo
        if tendencia > 0.5 and volatilidade < 0.3:
            return 20 if action == TradingStrategy.AGRESSIVA.value else -10

        # Em baixa tendência + alta vol: ser conservador
        if tendencia < -0.3 and volatilidade > 0.6:
            return 20 if action == TradingStrategy.CONSERVADORA.value else -10

        return 0


# ===== CONTROLLER: ESTRATÉGIA AGRESSIVA =====
@brl.problem(name="ControllerAgressivo")
class ControllerAgressivo:
    """
    Executa estratégia agressiva:
    - Alta exposição
    - Busca ativos com momentum
    - Stop loss largo
    """

    obs = brl.Dict(
        # Métricas agregadas do portfólio
        preco_medio=brl.Box(0, 500),
        retorno_1h_medio=brl.Box(-0.1, 0.1),
        retorno_24h_medio=brl.Box(-0.3, 0.3),

        # Momentum e volume médio
        rsi_medio=brl.Box(0, 100),  # Relative Strength Index médio
        volume_relativo_medio=brl.Box(0, 5),

        # Posições agregadas
        n_posicoes_long=brl.Box(0, 10),
        n_posicoes_short=brl.Box(0, 10),
        pnl_total=brl.Box(-50000, 50000),

        # Capital disponível
        capital_disponivel=brl.Box(0, 1000000),
        margem_disponivel=brl.Box(0, 1)
    )

    action = brl.Dict(
        # Ação de trading: -1=vender tudo, 0=manter, +1=comprar max
        acao_trading=brl.Box(-1, 1),

        # Stop loss (% de perda aceitável)
        stop_loss=brl.Box(0.05, 0.20),  # 5% a 20%

        # Take profit (% de ganho alvo)
        take_profit=brl.Box(0.10, 0.50),  # 10% a 50%

        # Tamanho da posição (% do capital por trade)
        tamanho_posicao=brl.Box(0.1, 0.5)  # 10% a 50%
    )

    objectives = brl.Terms(
        retorno=0.60,
        momentum=0.25,
        gestao_risco=0.15
    )

    def reward_retorno(self, state, action, next_state):
        """Maximiza retornos."""
        # PnL total
        pnl = next_state['pnl_total']
        return pnl / 100

    def reward_momentum(self, state, action, next_state):
        """Segue momentum."""
        # Alinha ação com RSI médio
        acao = action['acao_trading']
        rsi = state['rsi_medio']

        # Recompensa alinhar ação com momentum
        # RSI > 50 = momentum positivo, deve comprar
        momentum_signal = (rsi - 50) / 50  # -1 a +1
        alinhamento = acao * momentum_signal
        return alinhamento * 10

    def reward_gestao_risco(self, state, action, next_state):
        """Gestão de risco adequada."""
        # Penaliza stop loss muito apertado ou muito largo
        stop = action['stop_loss']
        if 0.08 <= stop <= 0.15:  # Range ideal
            return 10
        else:
            return -5


# ===== CONTROLLER: ESTRATÉGIA CONSERVADORA =====
@brl.problem(name="ControllerConservador")
class ControllerConservador:
    """
    Executa estratégia conservadora:
    - Baixa exposição
    - Foca em proteção de capital
    - Stop loss apertado
    """

    obs = brl.Dict(
        # Métricas de risco agregadas
        preco_medio=brl.Box(0, 500),
        volatilidade_media=brl.Box(0, 1),
        beta_portfolio=brl.Box(-2, 2),  # Beta vs mercado

        # Posições agregadas
        n_posicoes_total=brl.Box(0, 10),
        pnl_total=brl.Box(-50000, 50000),

        # Diversificação (simplificado)
        concentracao_portfolio=brl.Box(0, 1),  # 0=diverso, 1=concentrado

        capital_disponivel=brl.Box(0, 1000000)
    )

    action = brl.Dict(
        # Ação conservadora: menor range
        acao_trading=brl.Box(-0.5, 0.5),  # Menor range

        # Stop loss apertado
        stop_loss=brl.Box(0.03, 0.08),

        # Tamanho menor
        tamanho_posicao=brl.Box(0.05, 0.20)  # 5% a 20%
    )

    objectives = brl.Terms(
        preservacao_capital=0.50,
        diversificacao=0.30,
        retorno=0.20
    )

    def reward_preservacao_capital(self, state, action, next_state):
        """Evita perdas."""
        pnl = next_state['pnl_total']

        if pnl < 0:
            return pnl / 10  # Penaliza perdas fortemente
        else:
            return pnl / 50  # Retorno modesto ok

    def reward_diversificacao(self, state, action, next_state):
        """Incentiva diversificação."""
        # Usa concentração do portfólio (0=diverso, 1=concentrado)
        concentracao = state['concentracao_portfolio']

        # Menor concentração = melhor
        return (1 - concentracao) * 50

    def reward_retorno(self, state, action, next_state):
        """Retorno modesto."""
        pnl = next_state['pnl_total']
        return max(0, pnl / 100)  # Só recompensa ganhos


class HierarchicalTradingSystem:
    """Sistema de trading hierárquico."""

    def __init__(self):
        self.meta_controller = None
        self.controller_agressivo = None
        self.controller_conservador = None

        # Estado atual
        self.estrategia_atual = None
        self.tempo_na_estrategia = 0

        # Histórico
        self.historico = []

    def treinar_hierarquia(self):
        """Treina todos os níveis da hierarquia."""
        print("=" * 70)
        print("TREINAMENTO: HIERARCHICAL RL - TRADING SYSTEM")
        print("=" * 70)

        # 1. Treina Meta-Controller
        print("\n🎯 Treinando Meta-Controller (alto nível)...")
        problema_meta = MetaController()
        self.meta_controller = brl.train(
            problema_meta,
            algorithm='PPO',
            hours=0.5,
            config={'learning_rate': 3e-4, 'gamma': 0.99}
        )
        self.meta_controller.save('./modelos/meta_controller.pt')

        # 2. Treina Controller Agressivo
        print("\n⚡ Treinando Controller Agressivo...")
        problema_agressivo = ControllerAgressivo()
        self.controller_agressivo = brl.train(
            problema_agressivo,
            algorithm='PPO',
            hours=0.5,
            config={'learning_rate': 3e-4, 'gamma': 0.95}
        )
        self.controller_agressivo.save('./modelos/controller_agressivo.pt')

        # 3. Treina Controller Conservador
        print("\n🛡️  Treinando Controller Conservador...")
        problema_conservador = ControllerConservador()
        self.controller_conservador = brl.train(
            problema_conservador,
            algorithm='PPO',
            hours=0.5,
            config={'learning_rate': 3e-4, 'gamma': 0.95}
        )
        self.controller_conservador.save('./modelos/controller_conservador.pt')

        print("\n✅ Hierarquia completa treinada!")

    def carregar_hierarquia(self):
        """Carrega modelos treinados."""
        self.meta_controller = brl.load('./modelos/meta_controller.pt')
        self.controller_agressivo = brl.load('./modelos/controller_agressivo.pt')
        self.controller_conservador = brl.load('./modelos/controller_conservador.pt')

    def decidir_estrategia(self, estado_macro):
        """Meta-controller decide a estratégia."""
        decisao = self.meta_controller.decide(estado_macro, deterministic=True)
        self.estrategia_atual = TradingStrategy(decisao.action)
        self.tempo_na_estrategia = 0

        return self.estrategia_atual

    def executar_estrategia(self, estado_mercado):
        """Controller apropriado executa a estratégia."""
        if self.estrategia_atual == TradingStrategy.AGRESSIVA:
            decisao = self.controller_agressivo.decide(
                estado_mercado, deterministic=True
            )

        elif self.estrategia_atual == TradingStrategy.CONSERVADORA:
            decisao = self.controller_conservador.decide(
                estado_mercado, deterministic=True
            )

        elif self.estrategia_atual == TradingStrategy.MODERADA:
            # Mix de agressivo e conservador
            decisao_agr = self.controller_agressivo.decide(estado_mercado)
            decisao_cons = self.controller_conservador.decide(estado_mercado)

            # Média ponderada
            decisao = type('obj', (object,), {
                'action': {
                    'acao_trading': (decisao_agr.action['acao_trading'] * 0.5 +
                                    decisao_cons.action['acao_trading'] * 0.5),
                    'stop_loss': (decisao_agr.action['stop_loss'] +
                                 decisao_cons.action['stop_loss']) / 2,
                    'tamanho_posicao': (decisao_agr.action['tamanho_posicao'] +
                                       decisao_cons.action['tamanho_posicao']) / 2
                }
            })()

        else:  # LIQUIDACAO
            # Fecha todas as posições
            decisao = type('obj', (object,), {
                'action': {
                    'acao_trading': 0.0,
                    'stop_loss': 0.05,
                    'tamanho_posicao': 0.0
                }
            })()

        self.tempo_na_estrategia += 1
        return decisao


def demo_hierarchical_trading():
    """Demonstração do sistema hierárquico."""
    print("\n" + "=" * 70)
    print("DEMONSTRAÇÃO: TRADING HIERÁRQUICO")
    print("=" * 70)

    # Cria sistema
    sistema = HierarchicalTradingSystem()
    sistema.carregar_hierarquia()

    # Simula 10 ciclos de decisão
    print("\n📊 Simulando 10 ciclos de decisão...\n")

    for ciclo in range(10):
        print(f"\n{'='*70}")
        print(f"CICLO {ciclo + 1}")
        print(f"{'='*70}")

        # Estado macro (aleatório para demo)
        estado_macro = {
            'volatilidade_mercado': np.random.rand(),
            'tendencia_mercado': np.random.randn() * 0.5,
            'volume_mercado': np.random.rand(),
            'sp500_retorno_medio': np.random.randn() * 0.02,
            'nasdaq_retorno_medio': np.random.randn() * 0.025,
            'valor_portfolio': 100000,
            'exposicao_atual': np.random.rand() * 0.8,
            'posicoes_abertas': np.random.randint(5, 15),
            'retorno_7d': np.random.randn() * 0.1,
            'retorno_30d': np.random.randn() * 0.2,
            'sharpe_ratio': np.random.randn(),
            'max_drawdown': np.random.rand() * 0.15,
            'fear_greed_index': np.random.randint(20, 80),
            'put_call_ratio': 0.8 + np.random.rand() * 0.4,
            'hora_dia': np.random.randint(24),
            'dia_semana': np.random.randint(7),
            'fim_trimestre': 0,
            'margem_disponivel': np.random.rand(),
            'limite_risco_usado': np.random.rand() * 0.7
        }

        # Meta-controller decide estratégia
        estrategia = sistema.decidir_estrategia(estado_macro)
        print(f"\n🎯 ESTRATÉGIA ESCOLHIDA: {estrategia.name}")
        print(f"   Volatilidade: {estado_macro['volatilidade_mercado']:.2%}")
        print(f"   Tendência: {estado_macro['tendencia_mercado']:+.2%}")
        print(f"   Exposição atual: {estado_macro['exposicao_atual']:.2%}")

        # Estado do mercado (agregado)
        estado_mercado = {
            'preco_medio': np.random.rand() * 100 + 50,
            'retorno_1h_medio': np.random.randn() * 0.02,
            'retorno_24h_medio': np.random.randn() * 0.05,
            'rsi_medio': np.random.rand() * 100,
            'volume_relativo_medio': 0.5 + np.random.rand(),
            'n_posicoes_long': np.random.randint(0, 5),
            'n_posicoes_short': np.random.randint(0, 3),
            'pnl_total': np.random.randn() * 2500,
            'capital_disponivel': 100000,
            'margem_disponivel': 0.8,
            'volatilidade_media': np.random.rand() * 0.5,
            'beta_portfolio': np.random.randn() * 0.5 + 1,
            'concentracao_portfolio': np.random.rand() * 0.5,
            'n_posicoes_total': np.random.randint(1, 8)
        }

        # Executa estratégia
        decisao = sistema.executar_estrategia(estado_mercado)

        print(f"\n📈 AÇÃO EXECUTADA:")
        if 'acao_trading' in decisao.action:
            acao = decisao.action['acao_trading']
            if abs(acao) > 0.1:
                operacao = "COMPRA" if acao > 0 else "VENDA"
                intensidade = abs(acao)
                print(f"   {operacao} (intensidade: {intensidade:.2f})")
            else:
                print(f"   MANTER (neutro)")
        else:
            print(f"   Estratégia de LIQUIDAÇÃO")

        if hasattr(decisao.action, 'stop_loss'):
            print(f"\n   Stop Loss: {decisao.action['stop_loss']:.1%}")
            print(f"   Tamanho Posição: {decisao.action['tamanho_posicao']:.1%}")


if __name__ == "__main__":
    import os
    os.makedirs('./modelos', exist_ok=True)

    print("🚀 Iniciando exemplo Hierarchical RL\n")

    # Cria e treina sistema
    sistema = HierarchicalTradingSystem()
    sistema.treinar_hierarquia()

    # Demonstração
    demo_hierarchical_trading()

    print("\n" + "=" * 70)
    print("✅ Exemplo concluído!")
    print("=" * 70)
