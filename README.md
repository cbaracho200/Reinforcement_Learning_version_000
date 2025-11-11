# 🎯 COMO USAR O BUSINESS-RL

## 📚 Índice
1. [Como Desenvolver Modelos Passo a Passo](#-como-desenvolver-modelos-passo-a-passo)
2. [Instalação](#instalação)
3. [Primeiro Exemplo (Básico)](#primeiro-exemplo-básico)
4. [Exemplo Intermediário](#exemplo-intermediário)
5. [Exemplo Avançado](#exemplo-avançado)
6. [Problemas Pré-Construídos](#problemas-pré-construídos)
7. [API Completa](#api-completa)
8. [Dicas e Boas Práticas](#dicas-e-boas-práticas)

---

## 🎓 Como Desenvolver Modelos Passo a Passo

### Visão Geral do Processo

Desenvolver um modelo de Reinforcement Learning requer seguir um processo estruturado. Aqui está o guia completo:

```
┌─────────────────────────────────────────────────────────┐
│  1. Entender o Problema  →  2. Definir Observações     │
│           ↓                            ↓                 │
│  7. Refinar e Otimizar  ←  3. Definir Ações             │
│           ↑                            ↓                 │
│  6. Avaliar Resultados  ←  4. Criar Recompensas         │
│           ↑                            ↓                 │
│  5. Treinar o Modelo    ←  Definir Objetivos            │
└─────────────────────────────────────────────────────────┘
```

---

### 📋 Passo 1: Entender o Problema de Negócio

**Objetivo**: Definir claramente o que você quer otimizar.

**Perguntas a responder:**
- Qual decisão você precisa automatizar?
- Quais informações você tem disponíveis?
- Quais ações são possíveis?
- O que define sucesso?

**Exemplo prático:**

```python
"""
PROBLEMA: Sistema de precificação para e-commerce

DECISÃO: Que preço cobrar por produto a cada hora
INFORMAÇÕES: Estoque atual, demanda, preço concorrente, hora do dia
AÇÕES: Definir preço entre R$50 e R$500
SUCESSO: Maximizar receita mantendo competitividade
"""
```

---

### 🔭 Passo 2: Definir o Espaço de Observação

**Objetivo**: Especificar todas as informações que o agente pode "ver".

**Tipos de observação:**

```python
import business_rl as brl

# 1. VALORES CONTÍNUOS (números decimais)
obs_continuo = brl.Box(low=0, high=100)  # Ex: preço, temperatura

# 2. VALORES DISCRETOS (categorias)
obs_discreto = brl.Discrete(5)  # Ex: 5 categorias diferentes

# 3. MÚLTIPLAS OBSERVAÇÕES (dicionário)
obs_multiplo = brl.Dict(
    preco=brl.Box(0, 1000),        # Valor contínuo
    categoria=brl.Discrete(10),     # 10 categorias
    em_promocao=brl.Discrete(2)     # Sim/Não
)

# 4. OBSERVAÇÕES VETORIAIS (arrays)
obs_vetor = brl.Box(0, 1, shape=(10,))  # Array de 10 valores
```

**Exemplo completo:**

```python
@brl.problem(name="PrecificacaoDinamica")
class Precificacao:
    # O que o agente observa do ambiente
    obs = brl.Dict(
        # Preço atual do produto
        preco_atual=brl.Box(0, 1000),

        # Quantidade em estoque
        estoque=brl.Box(0, 500),

        # Preço do concorrente principal
        preco_concorrente=brl.Box(0, 1000),

        # Demanda nas últimas 24h
        demanda_24h=brl.Box(0, 1000),

        # Dia da semana (0=segunda, 6=domingo)
        dia_semana=brl.Discrete(7),

        # Hora do dia (0-23)
        hora=brl.Discrete(24),

        # Está em temporada alta?
        temporada_alta=brl.Discrete(2, labels=["nao", "sim"])
    )
```

**⚠️ Dicas importantes:**
- ✅ Inclua apenas informações relevantes para a decisão
- ✅ Normalize valores grandes (divida por um máximo)
- ✅ Use labels descritivos para variáveis discretas
- ❌ Evite incluir informações redundantes ou irrelevantes

---

### 🎮 Passo 3: Definir o Espaço de Ação

**Objetivo**: Especificar todas as ações que o agente pode tomar.

**Tipos de ação:**

```python
# 1. AÇÃO DISCRETA (escolher entre opções)
action = brl.Discrete(3, labels=["baixo", "medio", "alto"])

# 2. AÇÃO CONTÍNUA (valor numérico)
action = brl.Box(0, 1000)  # Ex: definir um preço

# 3. AÇÕES MÚLTIPLAS
action = brl.Dict(
    preco=brl.Box(0, 1000),
    desconto=brl.Box(0, 0.5),  # 0% a 50%
    promocao=brl.Discrete(2, labels=["nao", "sim"])
)

# 4. AÇÃO HÍBRIDA (discreta + contínua)
action = brl.Mixed(
    discreto=brl.Dict(
        estrategia=brl.Discrete(3, labels=["agressiva", "moderada", "conservadora"])
    ),
    continuo=brl.Dict(
        preco=brl.Box(50, 500),
        duracao_dias=brl.Box(1, 30)
    )
)
```

**Exemplo completo:**

```python
@brl.problem(name="PrecificacaoDinamica")
class Precificacao:
    obs = brl.Dict(...)  # Definido no Passo 2

    # Ação: ajustar preço e decidir sobre promoção
    action = brl.Dict(
        # Novo preço a cobrar
        preco=brl.Box(50, 500),

        # Percentual de desconto (0-30%)
        desconto=brl.Box(0, 0.30),

        # Destacar o produto?
        destaque=brl.Discrete(2, labels=["nao", "sim"])
    )
```

---

### 🎯 Passo 4: Criar Funções de Recompensa

**Objetivo**: Ensinar ao agente o que é "bom" ou "ruim".

**Princípios das recompensas:**
- Deve ser **mensurável** (retornar um número)
- Deve ser **frequente** (não apenas no final)
- Deve refletir o **objetivo real**

**Template básico:**

```python
def reward_nome(self, state, action, next_state):
    """
    Args:
        state: Estado antes da ação
        action: Ação tomada
        next_state: Estado depois da ação

    Returns:
        float: Valor da recompensa (maior = melhor)
    """
    # Seu cálculo aqui
    return recompensa
```

**Exemplos práticos:**

```python
@brl.problem(name="PrecificacaoDinamica")
class Precificacao:
    obs = brl.Dict(...)
    action = brl.Dict(...)

    # Recompensa 1: Maximizar receita
    def reward_receita(self, state, action, next_state):
        """Calcula a receita gerada pela decisão de preço."""
        # Estima vendas baseado no preço e desconto
        preco_final = action['preco'] * (1 - action['desconto'])

        # Modelo simples: quanto mais barato, mais vende
        # (você pode usar dados reais aqui)
        elasticidade = 2.0  # Sensibilidade ao preço
        demanda_base = state['demanda_24h']
        ratio_preco = preco_final / state['preco_concorrente']

        vendas_estimadas = demanda_base * (ratio_preco ** -elasticidade)
        vendas_reais = min(vendas_estimadas, state['estoque'])

        receita = preco_final * vendas_reais
        return receita

    # Recompensa 2: Manter competitividade
    def reward_competitividade(self, state, action, next_state):
        """Penaliza se ficar muito mais caro que concorrente."""
        preco_final = action['preco'] * (1 - action['desconto'])
        diferenca = preco_final - state['preco_concorrente']

        if diferenca > 100:  # Muito mais caro
            return -10
        elif diferenca < -50:  # Muito mais barato (perde margem)
            return -5
        else:
            return 0  # Preço competitivo

    # Recompensa 3: Evitar estoque zero
    def reward_estoque(self, state, action, next_state):
        """Penaliza se estoque ficar muito baixo."""
        if next_state['estoque'] < 10:
            return -20  # Penalidade grande
        elif next_state['estoque'] < 50:
            return -5   # Penalidade pequena
        else:
            return 0
```

**⚠️ Armadilhas comuns:**

```python
# ❌ RUIM: Recompensa muito esparsa
def reward_ruim(self, state, action, next_state):
    # Só dá recompensa no fim do mês
    if next_state['dia'] == 30:
        return calcular_lucro_mensal()
    return 0  # Nada nos outros dias (agente não aprende)

# ✅ BOM: Recompensa frequente
def reward_bom(self, state, action, next_state):
    # Recompensa a cada decisão
    return calcular_lucro_diario()

# ❌ RUIM: Recompensa não reflete objetivo
def reward_ruim(self, state, action, next_state):
    # Objetivo: maximizar lucro
    # Recompensa: número de vendas (ignora margem!)
    return action['vendas']

# ✅ BOM: Recompensa alinhada com objetivo
def reward_bom(self, state, action, next_state):
    receita = action['preco'] * action['vendas']
    custo = action['vendas'] * self.custo_unitario
    return receita - custo  # Lucro real
```

---

### 🎲 Passo 5: Definir Objetivos e Restrições

**Objetivo**: Combinar múltiplas recompensas e adicionar restrições.

#### 5.1 Múltiplos Objetivos

```python
@brl.problem(name="PrecificacaoDinamica")
class Precificacao:
    obs = brl.Dict(...)
    action = brl.Dict(...)

    # Combina múltiplas recompensas com pesos
    objectives = brl.Terms(
        receita=0.5,              # 50% do peso
        competitividade=0.3,      # 30% do peso
        estoque=0.2               # 20% do peso
    )

    # As funções de recompensa devem ter os mesmos nomes
    def reward_receita(self, state, action, next_state):
        ...

    def reward_competitividade(self, state, action, next_state):
        ...

    def reward_estoque(self, state, action, next_state):
        ...
```

#### 5.2 Adicionar Restrições

```python
@brl.problem(name="PrecificacaoDinamica")
class Precificacao:
    obs = brl.Dict(...)
    action = brl.Dict(...)
    objectives = brl.Terms(...)

    # Define limites que o agente deve respeitar
    constraints = {
        # Restrição HARD: nunca pode violar
        'preco_minimo': brl.Limit(
            func=lambda s, a: a['preco'],
            min_val=50,   # Preço não pode ser < R$50
            hard=True     # Ação inválida se violar
        ),

        # Restrição SOFT: pode violar mas recebe penalidade
        'margem_minima': brl.Limit(
            func=lambda s, a: a['preco'] - s['custo_unitario'],
            min_val=20,   # Margem mínima de R$20
            hard=False    # Pode violar mas é penalizado
        ),

        # Restrição de intervalo
        'desconto_maximo': brl.Limit(
            func=lambda s, a: a['desconto'],
            max_val=0.30,  # Máximo 30% de desconto
            hard=True
        )
    }
```

#### 5.3 Gestão de Risco (Opcional)

```python
@brl.problem(name="PrecificacaoDinamica")
class Precificacao:
    obs = brl.Dict(...)
    action = brl.Dict(...)
    objectives = brl.Terms(...)
    constraints = {...}

    # Considera os piores cenários
    risk = brl.CVaR(
        alpha=0.05,         # Considera 5% piores resultados
        max_drawdown=0.2    # Máxima perda aceitável de 20%
    )
```

---

### 🏋️ Passo 6: Testar e Validar o Problema

**Objetivo**: Garantir que sua definição está correta antes de treinar.

```python
import business_rl as brl

# 1. Crie o problema
problema = Precificacao()

# 2. Inspecione a definição
print("=" * 50)
print("INFORMAÇÕES DO PROBLEMA")
print("=" * 50)
print(problema.get_info())

# 3. Teste com dados de exemplo
estado_teste = {
    'preco_atual': 200,
    'estoque': 100,
    'preco_concorrente': 180,
    'demanda_24h': 50,
    'dia_semana': 0,  # Segunda
    'hora': 14,       # 14h
    'temporada_alta': 0  # Não
}

# 4. Teste uma ação de exemplo
acao_teste = {
    'preco': 190,
    'desconto': 0.10,  # 10%
    'destaque': 1      # Sim
}

# 5. Calcule as recompensas manualmente
print("\n" + "=" * 50)
print("TESTE DE RECOMPENSAS")
print("=" * 50)

# Simula próximo estado (normalmente vem do ambiente)
proximo_estado = estado_teste.copy()
proximo_estado['estoque'] = 90  # Vendeu 10 unidades

# Testa cada recompensa
r_receita = problema.reward_receita(estado_teste, acao_teste, proximo_estado)
r_comp = problema.reward_competitividade(estado_teste, acao_teste, proximo_estado)
r_est = problema.reward_estoque(estado_teste, acao_teste, proximo_estado)

print(f"Receita: {r_receita:.2f}")
print(f"Competitividade: {r_comp:.2f}")
print(f"Estoque: {r_est:.2f}")

# 6. Teste com treino rápido (6 minutos)
print("\n" + "=" * 50)
print("TESTE DE TREINO RÁPIDO")
print("=" * 50)

modelo = brl.train(problema, hours=0.1)  # 6 minutos
decisao = modelo.decide(estado_teste)

print(f"\nDecisão do modelo:")
print(f"  Preço: R$ {decisao.action['preco']:.2f}")
print(f"  Desconto: {decisao.action['desconto']*100:.1f}%")
print(f"  Destaque: {decisao.action['destaque']}")
print(f"  Confiança: {decisao.confidence:.2%}")
```

---

### 🚂 Passo 7: Treinar o Modelo

**Objetivo**: Treinar o agente com configurações adequadas.

#### 7.1 Treino Básico

```python
import business_rl as brl

problema = Precificacao()

# Treino simples (usa configurações padrão)
modelo = brl.train(problema, hours=1)

# Salvar o modelo
modelo.save('./modelos/precificacao_v1.pt')
```

#### 7.2 Treino Avançado

```python
# Mais controle sobre o processo
modelo = brl.train(
    problema,
    algorithm='PPO',      # Algoritmo (PPO ou SAC)
    hours=2,              # Tempo de treino
    config={
        'learning_rate': 3e-4,     # Taxa de aprendizado
        'batch_size': 256,         # Tamanho do lote
        'n_epochs': 10,            # Épocas por atualização
        'gamma': 0.99,             # Fator de desconto
        'gae_lambda': 0.95,        # GAE para vantagem
        'clip_range': 0.2,         # Clipping PPO
        'ent_coef': 0.01,          # Coeficiente de entropia
        'vf_coef': 0.5,            # Coeficiente de value function
    }
)
```

#### 7.3 Treino com Dashboard

```python
from business_rl.tools import TrainingDashboard

# Cria trainer
trainer = brl.Trainer(problema, algorithm='PPO')

# Inicia dashboard (abra http://localhost:5000 no navegador)
dashboard = TrainingDashboard(trainer, port=5000)
dashboard.start()

# Treina monitorando em tempo real
modelo = trainer.train(
    episodes=10000,           # Número de episódios
    save_path='./modelos/precificacao_v1.pt',
    checkpoint_freq=1000      # Salva a cada 1000 episódios
)
```

#### 7.4 Escolhendo o Algoritmo

```python
# PPO (Proximal Policy Optimization) - RECOMENDADO PARA INICIANTES
# ✅ Mais estável
# ✅ Funciona bem em vários problemas
# ✅ Bom para espaços discretos e contínuos
modelo_ppo = brl.train(problema, algorithm='PPO', hours=1)

# SAC (Soft Actor-Critic) - PARA AÇÕES CONTÍNUAS
# ✅ Melhor para ações contínuas complexas
# ✅ Mais exploração
# ⚠️ Pode ser mais lento
modelo_sac = brl.train(problema, algorithm='SAC', hours=2)
```

---

### 📊 Passo 8: Avaliar e Refinar

**Objetivo**: Validar o desempenho e iterar para melhorar.

#### 8.1 Teste com Dados Reais

```python
# Carrega modelo treinado
modelo = brl.load('./modelos/precificacao_v1.pt')

# Testa com múltiplos cenários
cenarios = [
    {
        'nome': 'Alta demanda',
        'estado': {'preco_atual': 200, 'estoque': 100, 'demanda_24h': 200, ...}
    },
    {
        'nome': 'Baixa demanda',
        'estado': {'preco_atual': 200, 'estoque': 100, 'demanda_24h': 20, ...}
    },
    {
        'nome': 'Estoque baixo',
        'estado': {'preco_atual': 200, 'estoque': 10, 'demanda_24h': 100, ...}
    }
]

print("=" * 60)
print("AVALIAÇÃO DO MODELO")
print("=" * 60)

for cenario in cenarios:
    decisao = modelo.decide(cenario['estado'], deterministic=True)

    print(f"\n{cenario['nome']}:")
    print(f"  Preço: R$ {decisao.action['preco']:.2f}")
    print(f"  Desconto: {decisao.action['desconto']*100:.1f}%")
    print(f"  Confiança: {decisao.confidence:.2%}")
```

#### 8.2 Comparar com Baseline

```python
# Crie uma política simples para comparação
def politica_simples(estado):
    """Sempre cobra 10% a menos que o concorrente."""
    return {
        'preco': estado['preco_concorrente'] * 0.9,
        'desconto': 0.0,
        'destaque': 0
    }

# Compare
estados_teste = [...]  # Seus dados de teste

receita_modelo = 0
receita_baseline = 0

for estado in estados_teste:
    # Decisão do modelo
    decisao_modelo = modelo.decide(estado)
    receita_modelo += simular_receita(estado, decisao_modelo.action)

    # Decisão baseline
    decisao_baseline = politica_simples(estado)
    receita_baseline += simular_receita(estado, decisao_baseline)

print(f"\nReceita Total:")
print(f"  Modelo RL: R$ {receita_modelo:,.2f}")
print(f"  Baseline:  R$ {receita_baseline:,.2f}")
print(f"  Melhoria:  {(receita_modelo/receita_baseline - 1)*100:.1f}%")
```

#### 8.3 Identificar Problemas Comuns

```python
# Problema 1: Modelo não aprende
# Solução: Verifique as recompensas
print("Recompensas médias por episódio:")
# Se sempre próximo de zero -> recompensas mal definidas

# Problema 2: Ações sempre iguais
# Solução: Aumente exploração
modelo = brl.train(problema, hours=1, config={
    'ent_coef': 0.1  # Aumenta entropia (exploração)
})

# Problema 3: Desempenho instável
# Solução: Reduza learning rate
modelo = brl.train(problema, hours=2, config={
    'learning_rate': 1e-4  # Menor que o padrão (3e-4)
})

# Problema 4: Viola restrições
# Solução: Torne restrições HARD
constraints = {
    'preco_minimo': brl.Limit(..., hard=True)  # Era False
}
```

#### 8.4 Iterar e Melhorar

```python
# VERSÃO 1: Modelo básico
modelo_v1 = brl.train(problema_v1, hours=1)
# Resultado: 70% de acurácia

# VERSÃO 2: Adiciona mais observações
problema_v2.obs = brl.Dict(
    # ... obs anteriores ...
    historico_vendas=brl.Box(0, 1000, shape=(7,))  # Últimos 7 dias
)
modelo_v2 = brl.train(problema_v2, hours=1.5)
# Resultado: 78% de acurácia

# VERSÃO 3: Refina recompensas
def reward_receita_v3(self, state, action, next_state):
    # Versão melhorada com modelo de demanda mais realista
    ...

modelo_v3 = brl.train(problema_v3, hours=2)
# Resultado: 85% de acurácia

# VERSÃO 4: Treina por mais tempo
modelo_v4 = brl.train(problema_v3, hours=5)
# Resultado: 90% de acurácia
```

---

### ✅ Checklist de Desenvolvimento

Use este checklist ao desenvolver seu modelo:

#### Fase 1: Definição
- [ ] Problema de negócio está claro
- [ ] Observações incluem todas as informações relevantes
- [ ] Ações representam todas as decisões possíveis
- [ ] Funções de recompensa refletem os objetivos reais

#### Fase 2: Validação
- [ ] `problema.get_info()` mostra informações corretas
- [ ] Testei recompensas manualmente com dados de exemplo
- [ ] Treino rápido (6 min) não dá erros
- [ ] Restrições estão bem definidas

#### Fase 3: Treino
- [ ] Escolhi o algoritmo apropriado (PPO ou SAC)
- [ ] Defini tempo de treino adequado (1-3h inicial)
- [ ] Configurei dashboard para monitoramento
- [ ] Salvei checkpoints durante o treino

#### Fase 4: Avaliação
- [ ] Testei com dados reais/realistas
- [ ] Comparei com baseline simples
- [ ] Modelo performa melhor que baseline
- [ ] Decisões fazem sentido intuitivamente

#### Fase 5: Produção
- [ ] Documentei versão e data do modelo
- [ ] Salvei configurações de treino
- [ ] Defini processo de re-treino
- [ ] Criei monitoramento de desempenho

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
