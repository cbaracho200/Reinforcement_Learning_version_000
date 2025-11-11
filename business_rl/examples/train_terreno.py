"""
Exemplo de treinamento para o problema de compra de terreno.
"""

import business_rl as brl


def main():
    # 1. Cria o problema
    problem = brl.CompraTerreno()
    
    # 2. Treina com configuração automática
    model = brl.train(
        problem,
        algorithm="auto",      # Seleciona automaticamente
        hours=0.5,            # 30 minutos de treino
        target_score=0.8      # Para quando atingir score
    )
    
    # 3. Testa o modelo
    test_state = {
        'taxa_juros': 0.12,
        'indice_demanda': 0.75,
        'velocidade_vendas': 0.6,
        'area_terreno': 5000,
        'coef_aproveitamento': 2.5,
        'preco_m2': 2000,
        'localizacao_score': 0.8,
        'caixa_disponivel': 30_000_000,
        'divida_atual': 10_000_000,
        'n_projetos_ativos': 4,
        'meses_ate_lancamento': 12,
        'concorrencia_proxima': 2
    }
    
    decision = model.decide(test_state)
    
    print(f"Decisão: {decision.action['decisao']}")
    print(f"Oferta: {decision.action['percentual_oferta']:.1%} do pedido")
    print(f"Permuta: {decision.action['percentual_permuta']:.1%}")
    print(f"Confiança: {decision.confidence:.1%}")


if __name__ == "__main__":
    main()