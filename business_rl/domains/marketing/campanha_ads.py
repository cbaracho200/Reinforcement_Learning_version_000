"""
Problema de otimização de campanhas de marketing digital.
"""

from ...tools.dsl import problem, Dict, Box, Discrete, Terms, Limit
import numpy as np


@problem(name="CampanhaAds", version="v1",
         description="Otimização de alocação de budget em campanhas digitais")
class CampanhaAds:
    """
    Otimização de campanhas considerando:
    - Múltiplos canais (Google, Meta, LinkedIn)
    - Segmentação de público
    - Budget constraints
    - ROAS targets
    """
    
    obs = Dict(
        # Performance atual
        ctr_google=Box(0.0, 0.1),          # Click-through rate
        ctr_meta=Box(0.0, 0.1),
        ctr_linkedin=Box(0.0, 0.1),
        
        cpc_google=Box(0.5, 50.0),         # Cost per click
        cpc_meta=Box(0.3, 30.0),
        cpc_linkedin=Box(2.0, 100.0),
        
        conv_rate_google=Box(0.0, 0.2),    # Conversion rate
        conv_rate_meta=Box(0.0, 0.2),
        conv_rate_linkedin=Box(0.0, 0.2),
        
        # Contexto de mercado
        dia_semana=Discrete(7),
        hora_dia=Discrete(24),
        sazonalidade=Box(0.5, 2.0),        # Multiplicador sazonal
        
        # Budget
        budget_restante=Box(0, 100000),     # R$ disponível
        dias_restantes=Box(1, 30),          # Dias de campanha
        
        # Competição
        competicao_google=Box(0.0, 1.0),
        competicao_meta=Box(0.0, 1.0),
        competicao_linkedin=Box(0.0, 1.0),
        
        # Qualidade
        quality_score_google=Box(1, 10),
        relevance_score_meta=Box(1, 10)
    )
    
    action = Dict(
        # Alocação de budget (%)
        alloc_google=Box(0.0, 1.0),
        alloc_meta=Box(0.0, 1.0),
        alloc_linkedin=Box(0.0, 1.0),
        
        # Estratégias de bid
        bid_strategy=Discrete(4, labels=[
            "max_clicks", "max_conversions", 
            "target_cpa", "target_roas"
        ]),
        
        # Ajustes de bid
        bid_adjustment=Box(0.5, 2.0),       # Multiplicador
        
        # Segmentação
        audience_expansion=Box(0.0, 1.0),   # Nível de expansão
        
        # Criativos
        n_creatives=Discrete(10)            # Número de variações
    )
    
    objectives = Terms(
        roas=0.4,           # Return on ad spend
        volume=0.3,         # Volume de conversões
        eficiencia=0.2,     # CPA baixo
        cobertura=0.1       # Alcance de mercado
    )
    
    constraints = {
        "budget_diario": Limit(
            lambda s: s['budget_restante'] / s['dias_restantes'],
            max_val=10000  # Max R$10k/dia
        ),
        "roas_minimo": Limit(
            lambda s: s.get('current_roas', 0),
            min_val=2.0  # ROAS mínimo de 2x
        ),
        "allocation_sum": Limit(
            lambda a: a['alloc_google'] + a['alloc_meta'] + a['alloc_linkedin'],
            max_val=1.0  # Soma das alocações <= 100%
        )
    }
    
    def reward_roas(self, state, action, next_state):
        """ROAS (Return on Ad Spend)."""
        
        # Calcula gasto por canal
        budget_dia = state['budget_restante'] / state['dias_restantes']
        
        spend_google = budget_dia * action['alloc_google']
        spend_meta = budget_dia * action['alloc_meta']
        spend_linkedin = budget_dia * action['alloc_linkedin']
        
        # Calcula receita esperada
        revenue_google = (spend_google / state['cpc_google']) * \
                        state['ctr_google'] * state['conv_rate_google'] * 500
        
        revenue_meta = (spend_meta / state['cpc_meta']) * \
                      state['ctr_meta'] * state['conv_rate_meta'] * 500
        
        revenue_linkedin = (spend_linkedin / state['cpc_linkedin']) * \
                          state['ctr_linkedin'] * state['conv_rate_linkedin'] * 800
        
        total_revenue = revenue_google + revenue_meta + revenue_linkedin
        total_spend = spend_google + spend_meta + spend_linkedin
        
        roas = total_revenue / max(total_spend, 1)
        
        return np.tanh(roas / 5)  # Normaliza com tanh
    
    def reward_volume(self, state, action, next_state):
        """Volume de conversões."""
        
        budget_dia = state['budget_restante'] / state['dias_restantes']
        
        # Estima conversões por canal
        conv_google = (budget_dia * action['alloc_google'] / state['cpc_google']) * \
                     state['ctr_google'] * state['conv_rate_google']
        
        conv_meta = (budget_dia * action['alloc_meta'] / state['cpc_meta']) * \
                   state['ctr_meta'] * state['conv_rate_meta']
        
        conv_linkedin = (budget_dia * action['alloc_linkedin'] / state['cpc_linkedin']) * \
                       state['ctr_linkedin'] * state['conv_rate_linkedin']
        
        total_conv = conv_google + conv_meta + conv_linkedin
        
        # Aplica sazonalidade
        total_conv *= state['sazonalidade']
        
        return np.log1p(total_conv) / 5
    
    def reward_eficiencia(self, state, action, next_state):
        """Eficiência (CPA baixo)."""
        
        # Similar ao ROAS mas focado em custo por aquisição
        budget_dia = state['budget_restante'] / state['dias_restantes']
        
        # Calcula CPA médio ponderado
        weights = [action['alloc_google'], action['alloc_meta'], action['alloc_linkedin']]
        cpcs = [state['cpc_google'], state['cpc_meta'], state['cpc_linkedin']]
        conv_rates = [state['conv_rate_google'], state['conv_rate_meta'], 
                     state['conv_rate_linkedin']]
        
        # CPA = CPC / conv_rate
        cpas = [cpc / max(cr, 0.001) for cpc, cr in zip(cpcs, conv_rates)]
        
        # Média ponderada
        avg_cpa = sum(w * cpa for w, cpa in zip(weights, cpas)) / max(sum(weights), 0.001)
        
        # Recompensa inversamente proporcional ao CPA
        return 100 / (avg_cpa + 1)
    
    def reward_cobertura(self, state, action, next_state):
        """Alcance de mercado."""
        
        # Recompensa diversificação
        allocations = [action['alloc_google'], action['alloc_meta'], action['alloc_linkedin']]
        
        # Entropia das alocações (máxima quando uniforme)
        entropy = 0
        for alloc in allocations:
            if alloc > 0:
                entropy -= alloc * np.log(alloc + 1e-8)
        
        # Normaliza
        max_entropy = -np.log(1/3) * 3
        
        return entropy / max_entropy