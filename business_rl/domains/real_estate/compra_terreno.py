"""
Problema de compra de terreno para incorporação imobiliária.
"""

from ...tools.dsl import (
    problem, Dict, Box, Discrete, Mixed, Terms, 
    Limit, CVaR, continuous, choices
)
import numpy as np


@problem(name="CompraTerreno", version="v1", 
         description="Decisão de compra de terreno com análise de viabilidade")
class CompraTerreno:
    """
    Problema de decisão para compra de terrenos considerando:
    - Análise de mercado
    - Restrições financeiras
    - Risco de liquidez
    - Potencial construtivo
    """
    
    # Observações do ambiente
    obs = Dict(
        # Mercado
        taxa_juros=Box(0.0, 0.20),           # Taxa de juros anual
        indice_demanda=Box(0.0, 1.0),        # Demanda normalizada
        velocidade_vendas=Box(0.0, 1.0),     # VSO do mercado
        
        # Terreno
        area_terreno=Box(100, 10000),        # m²
        coef_aproveitamento=Box(0.5, 4.0),   # CA permitido
        preco_m2=Box(500, 5000),             # R$/m²
        localizacao_score=Box(0.0, 1.0),     # Score de localização
        
        # Empresa
        caixa_disponivel=Box(0, 100_000_000),  # R$
        divida_atual=Box(0, 50_000_000),       # R$
        n_projetos_ativos=Discrete(10),        # Projetos em andamento
        
        # Timing
        meses_ate_lancamento=Box(3, 24),     # Meses estimados
        concorrencia_proxima=Box(0, 10)      # Lançamentos concorrentes
    )
    
    # Ações disponíveis
    action = Mixed(
        discreto=Dict(
            decisao=Discrete(3, labels=["passar", "analisar", "comprar"]),
            forma_pagamento=Discrete(3, labels=["vista", "prazo", "permuta"])
        ),
        continuo=Dict(
            percentual_oferta=Box(0.7, 1.1),    # % do preço pedido
            percentual_permuta=Box(0.0, 0.6),   # % em permuta
            prazo_pagamento=Box(3, 24)          # Meses para pagamento
        )
    )
    
    # Objetivos (multi-objetivo)
    objectives = Terms(
        lucro=0.4,        # Maximizar VPL
        liquidez=0.3,     # Manter liquidez saudável
        market_share=0.2, # Crescimento de mercado
        risco=0.1        # Minimizar risco (invertido)
    )
    
    # Restrições
    constraints = {
        "alavancagem": Limit(
            lambda s: s['divida_atual'] / max(s['caixa_disponivel'], 1),
            max_val=3.0,
            hard=False
        ),
        "caixa_minimo": Limit(
            lambda s: s['caixa_disponivel'],
            min_val=5_000_000,
            hard=True
        ),
        "exposicao_maxima": Limit(
            lambda s: s['n_projetos_ativos'],
            max_val=8,
            hard=False
        )
    }
    
    # Gestão de risco
    risk = CVaR(alpha=0.10, max_drawdown=0.30)
    
    # Métodos de recompensa para cada objetivo
    def reward_lucro(self, state, action, next_state):
        """Calcula recompensa de lucro (VPL estimado)."""
        
        if action['decisao'] != 2:  # Não comprou
            return 0.0
        
        # Parâmetros do terreno
        area = state['area_terreno']
        ca = state['coef_aproveitamento']
        preco_m2 = state['preco_m2']
        
        # Potencial construtivo
        area_construida = area * ca
        
        # Receita estimada
        preco_venda_m2 = 8000 * state['localizacao_score']
        vgv = area_construida * preco_venda_m2
        
        # Custo do terreno
        custo_terreno = area * preco_m2 * action['percentual_oferta']
        
        # Custo de construção
        custo_construcao = area_construida * 3000
        
        # Despesas
        despesas_total = (custo_terreno + custo_construcao) * 1.3
        
        # Lucro bruto
        lucro_bruto = vgv - despesas_total
        
        # Desconta pelo prazo
        taxa_desconto = state['taxa_juros']
        prazo = state['meses_ate_lancamento'] / 12
        vpl = lucro_bruto / ((1 + taxa_desconto) ** prazo)
        
        # Normaliza
        return vpl / 10_000_000
    
    def reward_liquidez(self, state, action, next_state):
        """Recompensa por manter liquidez."""
        
        caixa_ratio = next_state['caixa_disponivel'] / 20_000_000
        
        # Penaliza se caixa muito baixo
        if next_state['caixa_disponivel'] < 5_000_000:
            return -1.0
        
        # Recompensa logarítmica para caixa
        return np.log1p(caixa_ratio) / 3
    
    def reward_market_share(self, state, action, next_state):
        """Recompensa por crescimento."""
        
        if action['decisao'] != 2:
            return -0.1  # Penaliza levemente por não crescer
        
        # Recompensa baseada na qualidade da aquisição
        score = state['localizacao_score'] * state['indice_demanda']
        
        # Bonus por timing
        if state['concorrencia_proxima'] < 3:
            score *= 1.5
        
        return score
    
    def reward_risco(self, state, action, next_state):
        """Penalidade por risco excessivo."""
        
        # Calcula métricas de risco
        alavancagem = next_state['divida_atual'] / max(next_state['caixa_disponivel'], 1)
        exposicao = next_state['n_projetos_ativos'] / 10
        
        # Risco combinado
        risco = (alavancagem / 3) * 0.5 + exposicao * 0.5
        
        # Retorna negativo (queremos minimizar)
        return -risco