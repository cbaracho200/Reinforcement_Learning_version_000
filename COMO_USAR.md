# 🎯 COMO USAR O BUSINESS-RL

## 📚 Índice
1. [Instalação](#instalação)
2. [Primeiro Exemplo (Básico)](#primeiro-exemplo-básico)
3. [Exemplo Intermediário](#exemplo-intermediário)
4. [Exemplo Avançado](#exemplo-avançado)
5. [Problemas Pré-Construídos](#problemas-pré-construídos)
6. [API Completa](#api-completa)
7. [Dicas e Boas Práticas](#dicas-e-boas-práticas)

---

## 🔧 Instalação

### Opção 1: Instalação Local (Desenvolvimento)
```bash
cd C:\Users\incorporacao-04\Desktop\AGENTS_RL_AVANÇADOS\RL_001
pip install -e .
```

### Opção 2: Instalar dependências manualmente
```bash
pip install torch numpy gym flask
```

---

## 🌟 Primeiro Exemplo (Básico)

### Problema Simples: Decisão de Compra

```python
import business_rl as brl

# 1. Defina seu problema
@brl.problem(name="DecisaoCompra")
class DecisaoCompra:
    """Decidir se compra ou não baseado em preço e qualidade."""
    
    # O que você observa do ambiente
    obs = brl.Dict(
        preco=brl.Box(0, 1000),      # Preço do produto
        qualidade=brl.Box(0, 10)      # Qualidade (0-10)
    )
    
    # Ações disponíveis
    action = brl.Discrete(2, labels=["nao_comprar", "comprar"])
    
    # O que você quer maximizar
    objectives = brl.Terms(
        lucro=1.0  # Maximizar lucro
    )
    
    # Função de recompensa
    def reward_lucro(self, state, action, next_state):
        """Calcula o lucro da decisão."""
        if action == 1:  # comprou
            # Lucro = qualidade - custo
            return next_state['qualidade'] * 10 - next_state['preco']
        return 0  # não comprou

# 2. Crie o problema
problema = DecisaoCompra()

# 3. Treine o modelo (1 hora)
modelo = brl.train(problema, hours=1)

# 4. Use para tomar decisões
decisao = modelo.decide({
    "preco": 500,
    "qualidade": 8
})

print(f"Decisão: {decisao.action}")
print(f"Confiança: {decisao.confidence}")
```

---

## 🎯 Exemplo Intermediário

### Problema: Otimização de Campanha de Ads

```python
import business_rl as brl

@brl.problem(name="CampanhaAds")
class CampanhaAds:
    """Otimizar budget de campanha entre canais."""
    
    # Estado: métricas atuais
    obs = brl.Dict(
        budget_disponivel=brl.Box(0, 10000),
        ctr_facebook=brl.Box(0, 1),
        ctr_google=brl.Box(0, 1),
        conversoes_mes=brl.Box(0, 1000)
    )
    
    # Ação: quanto alocar para cada canal
    action = brl.Dict(
        facebook=brl.Box(0, 1),      # % do budget
        google=brl.Box(0, 1),        # % do budget
        instagram=brl.Box(0, 1)      # % do budget
    )
    
    # Múltiplos objetivos
    objectives = brl.Terms(
        roi=0.7,              # 70% peso no ROI
        conversoes=0.3        # 30% peso em conversões
    )
    
    # Restrições
    constraints = {
        'budget_total': brl.Limit(
            lambda s, a: a['facebook'] + a['google'] + a['instagram'],
            max_val=1.0,  # Soma não pode passar de 100%
            hard=True
        )
    }
    
    # Gestão de risco
    risk = brl.CVaR(
        alpha=0.05,           # 5% piores casos
        max_drawdown=0.2      # Máximo 20% de perda
    )
    
    def reward_roi(self, state, action, next_state):
        """ROI da alocação."""
        budget = state['budget_disponivel']
        gasto_fb = budget * action['facebook']
        gasto_gg = budget * action['google']
        gasto_ig = budget * action['instagram']
        
        # Simula receita (você substituiria por dados reais)
        receita = (gasto_fb * state['ctr_facebook'] * 50 +
                   gasto_gg * state['ctr_google'] * 45 +
                   gasto_ig * 0.03 * 40)
        
        gasto_total = gasto_fb + gasto_gg + gasto_ig
        return (receita - gasto_total) / (gasto_total + 1e-6)
    
    def reward_conversoes(self, state, action, next_state):
        """Número de conversões estimadas."""
        budget = state['budget_disponivel']
        return (budget * action['facebook'] * state['ctr_facebook'] * 0.02 +
                budget * action['google'] * state['ctr_google'] * 0.015)

# Treinar com configuração avançada
problema = CampanhaAds()

modelo = brl.train(
    problema,
    algorithm='PPO',          # Algoritmo
    hours=2,                  # 2 horas de treino
    config={
        'learning_rate': 3e-4,
        'batch_size': 256,
        'n_epochs': 10
    }
)

# Usar
estado_atual = {
    'budget_disponivel': 5000,
    'ctr_facebook': 0.05,
    'ctr_google': 0.08,
    'conversoes_mes': 150
}

decisao = modelo.decide(estado_atual)
print(f"Alocação recomendada:")
print(f"  Facebook: {decisao.action['facebook']*100:.1f}%")
print(f"  Google: {decisao.action['google']*100:.1f}%")
print(f"  Instagram: {decisao.action['instagram']*100:.1f}%")
```

---

## 🏢 Exemplo Avançado: Compra de Terreno

### Usando o Problema Pré-Construído

```python
import business_rl as brl
from business_rl.domains.real_estate import CompraTerreno

# 1. Usa problema pré-construído
problema = CompraTerreno()

# 2. Treina com dashboard em tempo real
trainer = brl.Trainer(problema, algorithm='PPO')

# Abre dashboard no navegador
dashboard = brl.TrainingDashboard(trainer, port=5000)
dashboard.start()

# Treina (acompanhe no navegador em http://localhost:5000)
modelo = trainer.train(hours=3)

# 3. Avalia um terreno específico
terreno = {
    'preco_m2': 500,
    'area_total': 1000,
    'zoneamento': 'residencial',
    'acesso_agua': True,
    'acesso_luz': True,
    'distancia_centro': 5.0,
    'valorizacao_historica': 0.08
}

decisao = modelo.decide(terreno)
print(f"Decisão: {decisao.action}")
print(f"Valor estimado: ${decisao.value:.2f}")
print(f"Confiança: {decisao.confidence:.2%}")
```

---

## 📦 Problemas Pré-Construídos

### 1. Compra de Terreno

```python
from business_rl.domains.real_estate import CompraTerreno

problema = CompraTerreno()
modelo = brl.train(problema, hours=1)

decisao = modelo.decide({
    'preco_m2': 500,
    'area_total': 1000,
    'zoneamento': 'residencial',
    # ... outros campos
})
```

### 2. Campanha de Ads

```python
from business_rl.domains.marketing import CampanhaAds

problema = CampanhaAds()
modelo = brl.train(problema, hours=1)

decisao = modelo.decide({
    'budget_disponivel': 5000,
    'ctr_facebook': 0.05,
    # ... outros campos
})
```

---

## 🔍 API Completa

### 1. Definir Observações

```python
# Observação simples (número)
obs = brl.Box(0, 100)

# Observações múltiplas (dicionário)
obs = brl.Dict(
    preco=brl.Box(0, 1000),
    quantidade=brl.Box(0, 100),
    categoria=brl.Discrete(5)  # 5 categorias
)
```

### 2. Definir Ações

```python
# Ação discreta simples
action = brl.Discrete(3, labels=["baixo", "medio", "alto"])

# Ação contínua
action = brl.Box(0, 1)

# Ações múltiplas
action = brl.Dict(
    preco=brl.Box(0, 1000),
    promocao=brl.Discrete(2, labels=["sim", "nao"])
)

# Ações híbridas (discretas + contínuas)
action = brl.Mixed(
    discreto=brl.Dict(
        tipo_campanha=brl.Discrete(3, labels=["agressiva", "moderada", "conservadora"])
    ),
    continuo=brl.Dict(
        budget=brl.Box(0, 10000),
        duracao_dias=brl.Box(1, 30)
    )
)
```

### 3. Definir Objetivos

```python
# Objetivo simples
objectives = brl.Terms(lucro=1.0)

# Múltiplos objetivos com pesos
objectives = brl.Terms(
    lucro=0.6,
    satisfacao_cliente=0.3,
    impacto_ambiental=0.1
)
```

### 4. Definir Restrições

```python
constraints = {
    'budget': brl.Limit(
        func=lambda s, a: a['gasto_total'],
        max_val=10000,
        hard=True  # Nunca pode violar
    ),
    'tempo': brl.Limit(
        func=lambda s, a: a['horas_trabalho'],
        min_val=8,
        max_val=40,
        hard=False  # Pode violar com penalidade
    )
}
```

### 5. Gestão de Risco

```python
risk = brl.CVaR(
    alpha=0.05,           # Considera 5% piores cenários
    max_drawdown=0.2      # Máximo 20% de perda aceitável
)
```

### 6. Treinar

```python
# Básico
modelo = brl.train(problema, hours=1)

# Avançado
modelo = brl.train(
    problema,
    algorithm='PPO',  # ou 'SAC'
    hours=2,
    config={
        'learning_rate': 3e-4,
        'batch_size': 256,
        'gamma': 0.99,
        'n_epochs': 10
    }
)

# Com Trainer (mais controle)
trainer = brl.Trainer(problema, algorithm='PPO')
modelo = trainer.train(
    episodes=1000,
    save_path='./modelos/meu_modelo.pt'
)
```

### 7. Usar Modelo

```python
# Decisão determinística
decisao = modelo.decide(estado, deterministic=True)

# Decisão com exploração
decisao = modelo.decide(estado, deterministic=False)

# Acessar informações da decisão
print(decisao.action)       # Ação escolhida
print(decisao.value)        # Valor esperado
print(decisao.confidence)   # Confiança (0-1)
print(decisao.log_prob)     # Log-probabilidade
print(decisao.entropy)      # Entropia (exploração)
```

---

## 💡 Dicas e Boas Práticas

### 1. Começar Simples
```python
# ✅ BOM: Comece com problema simples
@brl.problem(name="Simples")
class Simples:
    obs = brl.Box(0, 100)
    action = brl.Discrete(2)
    objectives = brl.Terms(lucro=1.0)

# ❌ EVITE: Começar com muita complexidade
```

### 2. Normalizar Observações
```python
# ✅ BOM: Valores normalizados (0-1)
obs = brl.Dict(
    preco_normalizado=brl.Box(0, 1),  # Dividiu por max
    quantidade_normalizada=brl.Box(0, 1)
)

# ❌ EVITE: Escalas muito diferentes
obs = brl.Dict(
    preco=brl.Box(0, 1000000),  # Muito grande
    quantidade=brl.Box(0, 10)    # Muito pequena
)
```

### 3. Função de Recompensa Clara
```python
# ✅ BOM: Recompensa clara e mensurável
def reward_lucro(self, state, action, next_state):
    receita = action['preco'] * action['quantidade']
    custo = action['quantidade'] * 10
    return receita - custo

# ❌ EVITE: Recompensa complexa demais
def reward_lucro(self, state, action, next_state):
    # Muitas condições, difícil de aprender
    if state['dia'] == 'segunda' and action > 5:
        if next_state['estoque'] < 100:
            return math.log(action) * state['preco'] ** 2
    # ...
```

### 4. Testar Incrementalmente
```python
# 1. Teste o problema
problema = MeuProblema()
print(problema.get_info())

# 2. Teste com poucos episódios
modelo = brl.train(problema, hours=0.1)  # 6 minutos

# 3. Teste decisão
decisao = modelo.decide(estado_teste)
print(decisao)

# 4. Se funcionar, aumente o treino
modelo = brl.train(problema, hours=1)
```

### 5. Usar Dashboard
```python
# Monitore o treino em tempo real
from business_rl.tools import TrainingDashboard

trainer = brl.Trainer(problema)
dashboard = TrainingDashboard(trainer, port=5000)
dashboard.start()

# Abra http://localhost:5000 no navegador
modelo = trainer.train(hours=2)
```

---

## 🎓 Exemplos de Casos de Uso

### E-commerce: Precificação Dinâmica
```python
@brl.problem(name="PrecificacaoDinamica")
class Precificacao:
    obs = brl.Dict(
        demanda_atual=brl.Box(0, 1000),
        estoque=brl.Box(0, 500),
        preco_concorrente=brl.Box(0, 200),
        dia_semana=brl.Discrete(7)
    )
    
    action = brl.Box(0, 200)  # Preço a cobrar
    
    objectives = brl.Terms(
        receita=0.8,
        market_share=0.2
    )
```

### Logística: Roteamento de Entregas
```python
@brl.problem(name="Roteamento")
class Roteamento:
    obs = brl.Dict(
        localizacao_atual=brl.Box(-180, 180, shape=(2,)),  # lat, lon
        entregas_pendentes=brl.Box(0, 50),
        trafego=brl.Box(0, 1),
        combustivel=brl.Box(0, 100)
    )
    
    action = brl.Discrete(10)  # Próxima entrega (top 10)
    
    objectives = brl.Terms(
        tempo=0.5,
        custo=0.3,
        satisfacao=0.2
    )
```

### Finanças: Portfolio Management
```python
@brl.problem(name="Portfolio")
class Portfolio:
    obs = brl.Dict(
        precos_acoes=brl.Box(0, 1000, shape=(10,)),  # 10 ações
        portfolio_atual=brl.Box(0, 1, shape=(10,)),   # % alocado
        volatilidade=brl.Box(0, 1, shape=(10,))
    )
    
    action = brl.Box(0, 1, shape=(10,))  # Nova alocação
    
    objectives = brl.Terms(retorno=0.7, risco=0.3)
    
    risk = brl.CVaR(alpha=0.05, max_drawdown=0.15)
```

---

## 📞 Precisa de Ajuda?

- 📚 **Documentação**: Veja exemplos em `business_rl/examples/`
- 🐛 **Problemas**: Verifique `log.txt`
- 💡 **Dúvidas**: Consulte este guia

---

## 🚀 Próximos Passos

1. ✅ Execute `executar_validacao.bat` para garantir que tudo está funcionando
2. 📝 Copie um dos exemplos acima
3. ✏️ Adapte para seu problema
4. 🏃 Execute e experimente!

**Boa sorte! 🎉**
