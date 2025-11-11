# Business-RL 🚀

Framework de Reinforcement Learning focado em decisões empresariais complexas.

## ✨ Features

- **DSL Intuitivo**: Defina problemas de negócio sem conhecer RL profundamente
- **Multi-objetivo**: Otimize múltiplas métricas simultaneamente
- **Gestão de Risco**: CVaR, drawdown e métricas de cauda integradas
- **Restrições**: Suporte nativo para constraints de negócio
- **Dashboard**: Monitoramento em tempo real do treinamento
- **Production-Ready**: Export direto para APIs e deploy

## 📦 Instalação
```bash
pip install business-rl
```

## 🚀 Quick Start
```python
import business_rl as brl

# Defina seu problema
@brl.problem(name="MeuProblema")
class MeuProblema:
    obs = brl.Dict(
        metrica1=brl.Box(0, 100),
        metrica2=brl.Box(0, 1)
    )
    
    action = brl.Discrete(3, labels=["opcao1", "opcao2", "opcao3"])
    
    objectives = brl.Terms(
        lucro=0.7,
        satisfacao=0.3
    )

# Treine
problema = MeuProblema()
modelo = brl.train(problema, hours=1)

# Use
decisao = modelo.decide({"metrica1": 50, "metrica2": 0.7})
print(f"Ação recomendada: {decisao.action}")
```

## 📊 Problemas Pré-Construídos

- `CompraTerreno`: Decisão de compra de terrenos
- `CampanhaAds`: Otimização de campanhas digitais
- `InventoryManagement`: Gestão de estoque
- `PricingOptimization`: Precificação dinâmica

## 🎯 Casos de Uso

- **Real Estate**: Análise de viabilidade e timing de compra
- **Marketing**: Alocação de budget e otimização de campanhas
- **Finance**: Gestão de portfolio e risco
- **Operations**: Cadeia de suprimentos e logística

## 📚 Documentação

Visite [nossa documentação](https://business-rl.readthedocs.io) para guias detalhados.

## 🤝 Contribuindo

Contribuições são bem-vindas! Veja [CONTRIBUTING.md](CONTRIBUTING.md) para detalhes.

## 📄 Licença

MIT License - veja [LICENSE](LICENSE) para detalhes.