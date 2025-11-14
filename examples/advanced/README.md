# 🎓 Exemplos Avançados - Business RL

Esta pasta contém exemplos avançados e completos de uso do framework Business-RL.

## 📋 Exemplos Disponíveis

### 1. 💼 Gestão de Portfólio (`01_portfolio_management.py`)

**Problema:** Otimizar alocação de capital em 5 ativos diferentes

**Demonstra:**
- ✅ Múltiplas observações (preços, volatilidade, retornos históricos)
- ✅ Ações contínuas (percentual de alocação)
- ✅ Múltiplos objetivos (retorno vs risco)
- ✅ Restrições complexas (limites de alocação)
- ✅ Gestão de risco com CVaR
- ✅ Algoritmo SAC para ações contínuas

**Executar:**
```bash
python examples/advanced/01_portfolio_management.py
```

**O que você vai aprender:**
- Como modelar problemas financeiros
- Usar CVaR para gestão de risco
- Trabalhar com espaços de ação contínuos
- Comparar com estratégias baseline

---

### 2. 💰 Precificação Dinâmica (`02_dynamic_pricing.py`)

**Problema:** Definir preço ótimo para maximizar lucro em e-commerce

**Demonstra:**
- ✅ Observações temporais (hora, dia, sazonalidade)
- ✅ Ações híbridas (preço + desconto + promoções)
- ✅ Modelagem de elasticidade de demanda
- ✅ Competição com concorrentes
- ✅ Trade-off entre margem e volume

**Executar:**
```bash
python examples/advanced/02_dynamic_pricing.py
```

**O que você vai aprender:**
- Modelar demanda elástica
- Combinar ações discretas e contínuas
- Considerar sazonalidade
- Otimizar múltiplos objetivos conflitantes

---

### 3. 📦 Gestão de Estoque (`03_inventory_management.py`)

**Problema:** Gerenciar estoque de múltiplos produtos

**Demonstra:**
- ✅ Múltiplos produtos simultâneos
- ✅ Restrições de orçamento e capacidade
- ✅ Previsão de demanda
- ✅ Lead time de fornecedores
- ✅ Minimização de rupturas e excessos

**Executar:**
```bash
python examples/advanced/03_inventory_management.py
```

**O que você vai aprender:**
- Gerenciar múltiplos produtos
- Trabalhar com restrições dinâmicas
- Otimizar capital de giro
- Evitar obsolescência

---

### 4. 🎯 Contextual Bandits (`04_contextual_bandits_personalization.py`)

**Problema:** Personalização de conteúdo em tempo real

**Demonstra:**
- ✅ Contextual Multi-Armed Bandits
- ✅ Exploration vs Exploitation (Thompson Sampling, UCB)
- ✅ Ensemble de estratégias (Thompson, UCB, Epsilon-Greedy, RL)
- ✅ A/B testing inteligente
- ✅ Cold-start problem

**Executar:**
```bash
python examples/advanced/04_contextual_bandits_personalization.py
```

**O que você vai aprender:**
- Implementar bandits contextuais
- Combinar múltiplas estratégias de exploração
- Realizar A/B testing adaptativo
- Personalizar conteúdo por usuário
- Comparar Thompson Sampling vs UCB vs RL

**Casos de uso:**
- Recomendação de produtos
- Personalização de emails/banners
- Seleção de conteúdo dinâmico

---

### 5. 🤝 Multi-Agent RL (`05_multi_agent_supply_chain.py`)

**Problema:** Cadeia de suprimentos com múltiplos agentes cooperando

**Demonstra:**
- ✅ Multi-Agent Reinforcement Learning (MARL)
- ✅ Comunicação entre agentes
- ✅ 3 agentes cooperando (Fornecedor, Distribuidor, Varejista)
- ✅ Coordenação e troca de mensagens
- ✅ Análise de Bullwhip Effect

**Executar:**
```bash
python examples/advanced/05_multi_agent_supply_chain.py
```

**O que você vai aprender:**
- Treinar múltiplos agentes simultaneamente
- Implementar comunicação entre agentes
- Coordenar decisões distribuídas
- Minimizar efeito cascata (Bullwhip)
- Credit assignment em MARL

**Casos de uso:**
- Supply chain management
- Sistemas distribuídos
- Leilões multi-agente
- Jogos cooperativos

---

### 6. 🏗️ Hierarchical RL (`06_hierarchical_trading_system.py`)

**Problema:** Sistema de trading com decisões hierárquicas

**Demonstra:**
- ✅ Hierarchical Reinforcement Learning (HRL)
- ✅ Meta-Controller (estratégia de alto nível)
- ✅ Controllers (execução tática)
- ✅ Temporal abstractions (opções/skills)
- ✅ Decomposição hierárquica de problemas complexos

**Estrutura:**
```
Meta-Controller → Escolhe ESTRATÉGIA (Agressiva/Moderada/Conservadora)
      ↓
Controllers → Executam AÇÕES específicas (compra/venda, stop loss)
```

**Executar:**
```bash
python examples/advanced/06_hierarchical_trading_system.py
```

**O que você vai aprender:**
- Decompor problemas complexos em hierarquias
- Treinar políticas de alto e baixo nível
- Usar temporal abstractions
- Adaptar estratégias ao contexto

**Casos de uso:**
- Trading automatizado
- Robótica (planejamento hierárquico)
- Jogos complexos
- Navegação autônoma

---

### 7. 🎲 Ensemble Learning (`07_ensemble_model_selection.py`)

**Problema:** Combinar múltiplos agentes RL para robustez

**Demonstra:**
- ✅ Ensemble de agentes com diferentes configurações
- ✅ Voting (uniforme e ponderado)
- ✅ Stacking (meta-learning)
- ✅ Mixture of Experts
- ✅ Dynamic model selection
- ✅ Análise de diversidade

**Técnicas:**
1. **Voting Ensemble:** voto majoritário/ponderado
2. **Stacking:** meta-modelo aprende a combinar
3. **Dynamic Selection:** escolhe melhor modelo por contexto
4. **Mixture of Experts:** combina especializações

**Executar:**
```bash
python examples/advanced/07_ensemble_model_selection.py
```

**O que você vai aprender:**
- Treinar ensemble de agentes
- Combinar predições de múltiplos modelos
- Usar meta-learning para seleção
- Avaliar diversidade do ensemble
- Aumentar robustez via ensemble

**Casos de uso:**
- Prevenção de churn
- Detecção de fraude
- Sistemas críticos (saúde, finanças)
- Decisões de alto risco

---

## 🚀 Como Executar os Exemplos

### Pré-requisitos

1. **Instalar o Business-RL:**
```bash
pip install git+https://github.com/cbaracho200/Reinforcement_Learning_version_000.git
```

2. **Ou instalar localmente:**
```bash
git clone https://github.com/cbaracho200/Reinforcement_Learning_version_000.git
cd Reinforcement_Learning_version_000
pip install -e .
```

### Executar um exemplo

```bash
# Navegar até a pasta do projeto
cd Reinforcement_Learning_version_000

# Exemplos básicos
python examples/advanced/01_portfolio_management.py
python examples/advanced/02_dynamic_pricing.py
python examples/advanced/03_inventory_management.py

# Exemplos avançados (técnicas modernas)
python examples/advanced/04_contextual_bandits_personalization.py
python examples/advanced/05_multi_agent_supply_chain.py
python examples/advanced/06_hierarchical_trading_system.py
python examples/advanced/07_ensemble_model_selection.py
```

### Ajustar tempo de treino

Por padrão, os exemplos treinam por 2 horas. Para testes rápidos, edite o arquivo e mude:

```python
# De:
modelo = brl.train(problema, hours=2)

# Para (teste rápido - 10 minutos):
modelo = brl.train(problema, hours=0.17)
```

---

## 📊 Estrutura dos Exemplos

Todos os exemplos seguem a mesma estrutura:

```python
# 1. Definição do problema
@brl.problem(name="NomeProblema")
class MeuProblema:
    obs = brl.Dict(...)      # Observações
    action = brl.Dict(...)   # Ações
    objectives = brl.Terms(...)  # Objetivos
    constraints = {...}      # Restrições
    risk = brl.CVaR(...)    # Gestão de risco (opcional)

    # Funções de recompensa
    def reward_objetivo1(self, state, action, next_state):
        ...

    def reward_objetivo2(self, state, action, next_state):
        ...

# 2. Função de treino
def treinar_modelo():
    problema = MeuProblema()
    modelo = brl.train(problema, algorithm='PPO', hours=2)
    modelo.save('./modelos/meu_modelo.pt')
    return modelo

# 3. Função de teste
def testar_modelo():
    modelo = brl.load('./modelos/meu_modelo.pt')
    # Testa com vários cenários
    ...

# 4. Execução
if __name__ == "__main__":
    modelo = treinar_modelo()
    testar_modelo()
```

---

## 🎯 Conceitos Demonstrados

### 1. Tipos de Observação
- **Contínuas:** `brl.Box(0, 100)` - valores numéricos
- **Discretas:** `brl.Discrete(5)` - categorias
- **Vetoriais:** `brl.Box(0, 1, shape=(10,))` - arrays
- **Dicionários:** `brl.Dict(...)` - múltiplas observações

### 2. Tipos de Ação
- **Discretas:** escolher entre opções
- **Contínuas:** valores numéricos
- **Híbridas:** combinação de discretas e contínuas

### 3. Restrições
- **Hard:** nunca pode violar (ação inválida)
- **Soft:** pode violar com penalidade
- **Dinâmicas:** dependem do estado

### 4. Algoritmos
- **PPO:** recomendado para iniciantes, estável
- **SAC:** melhor para ações contínuas complexas

### 5. Gestão de Risco
- **CVaR:** considera piores cenários
- **Max Drawdown:** limita perdas máximas

### 6. Técnicas Avançadas ⭐ NOVO

#### Contextual Bandits
- **Exploration vs Exploitation:** balanço entre explorar e exploitar
- **Thompson Sampling:** amostragem bayesiana
- **UCB (Upper Confidence Bound):** intervalos de confiança
- **Epsilon-Greedy:** exploração aleatória

#### Multi-Agent RL
- **Comunicação entre agentes:** troca de mensagens
- **Coordenação descentralizada:** decisões distribuídas
- **Credit assignment:** atribuir recompensa a agentes
- **Emergência:** comportamento cooperativo emergente

#### Hierarchical RL
- **Meta-Controller:** decisões estratégicas de alto nível
- **Controllers:** execução tática de baixo nível
- **Temporal abstractions:** skills e opções reutilizáveis
- **Decomposição hierárquica:** dividir problemas complexos

#### Ensemble Learning
- **Voting:** combinação por votação
- **Stacking:** meta-modelo aprende a combinar
- **Mixture of Experts:** especialistas para diferentes contextos
- **Dynamic Selection:** escolhe modelo por situação
- **Diversidade:** importância de modelos diferentes

---

## 🆚 Quando Usar Cada Técnica

| Técnica | Quando Usar | Exemplo |
|---------|-------------|---------|
| **Contextual Bandits** | Decisões independentes, feedback imediato | Recomendação de produtos, A/B testing |
| **Multi-Agent RL** | Múltiplos tomadores de decisão cooperando | Supply chain, leilões, jogos |
| **Hierarchical RL** | Problemas com múltiplos níveis de abstração | Trading, robótica, planejamento |
| **Ensemble** | Aumentar robustez e performance | Sistemas críticos, detecção de fraude |

---

## 💡 Dicas para Usar os Exemplos

### 1. Começar Simples
```bash
# Execute primeiro o exemplo mais simples
python examples/advanced/01_portfolio_management.py
```

### 2. Experimentar com Parâmetros
```python
# Teste diferentes configurações
modelo = brl.train(
    problema,
    algorithm='PPO',
    hours=1,  # Reduza para testes
    config={
        'learning_rate': 3e-4,  # Ajuste conforme necessário
        'batch_size': 128,      # Reduza se tiver pouca memória
    }
)
```

### 3. Criar Seus Próprios Cenários
```python
# Adicione seus próprios casos de teste
cenarios = [
    {
        'nome': 'Meu Cenário',
        'estado': {
            # Seus dados aqui
        }
    }
]
```

### 4. Comparar com Baseline
```python
# Sempre compare com uma estratégia simples
def estrategia_simples(estado):
    return acao_padrao

# Compare resultados
resultado_rl = testar_modelo_rl()
resultado_baseline = testar_baseline()
print(f"Melhoria: {(resultado_rl/resultado_baseline - 1)*100:.1f}%")
```

---

## 🔧 Troubleshooting

### Erro: "Módulo business_rl não encontrado"
```bash
# Certifique-se de ter instalado
pip install git+https://github.com/cbaracho200/Reinforcement_Learning_version_000.git
```

### Treino muito lento
```python
# Reduza o tempo de treino
modelo = brl.train(problema, hours=0.5)  # 30 minutos

# Ou reduza batch_size
config={'batch_size': 64}
```

### Modelo não aprende
```python
# 1. Verifique as recompensas
print(problema.reward_objetivo1(estado, acao, proximo_estado))

# 2. Aumente exploração
config={'ent_coef': 0.1}

# 3. Reduza learning rate
config={'learning_rate': 1e-4}
```

---

## 📚 Próximos Passos

1. ✅ Execute todos os exemplos
2. 📝 Modifique um exemplo para seu caso de uso
3. 🎯 Crie seu próprio problema do zero
4. 🚀 Compartilhe seus resultados!

---

## 🤝 Contribuindo

Tem um exemplo interessante? Contribua!

1. Fork o repositório
2. Crie seu exemplo em `examples/advanced/`
3. Siga a estrutura dos exemplos existentes
4. Envie um Pull Request

---

## 📞 Precisa de Ajuda?

- 📚 Veja a documentação principal no `README.md`
- 💡 Consulte a seção "Como Desenvolver Modelos Passo a Passo"
- 🐛 Reporte problemas no GitHub Issues

**Boa sorte! 🎉**
