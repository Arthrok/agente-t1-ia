import numpy as np
import random
import matplotlib.pyplot as plt  # Para criar os gráficos no final

# Classe que define o ambiente da casa inteligente
class SmartHomeEnvironment:
    def __init__(self):
        # Estado inicial do ambiente (temperatura, tarifa de energia, presença)
        self.temperature = random.choice(["baixa", "média", "alta"])  # Temperatura atual
        self.tariff = random.choice(["baixa", "média", "alta"])       # Tarifa de energia atual
        self.presence = random.choice([True, False])                  # Presença de moradores
    
    def get_state(self):
        # Retorna o estado atual do ambiente como uma tupla
        return (self.temperature, self.tariff, self.presence)
    
    def update_environment(self):
        # Atualiza o ambiente com novos valores aleatórios
        self.temperature = random.choice(["baixa", "média", "alta"])
        self.tariff = random.choice(["baixa", "média", "alta"])
        self.presence = random.choice([True, False])
    
    def get_reward(self, action):
        # Define as recompensas com base na ação e no estado atual do ambiente
        if action == "ligar_AC" and self.temperature == "alta" and self.presence:
            return 10  # Alta recompensa por manter o conforto
        elif action == "desligar_AC" and self.temperature != "alta":
            return 5  # Boa economia de energia
        elif action == "ligar_Luzes" and self.presence:
            return 2  # Luzes ligadas quando necessário
        elif action == "desligar_Luzes" and not self.presence:
            return 3  # Economia de energia
        else:
            return -5  # Penalidade para ações inadequadas

# Classe que define o agente que aprende com Q-Learning
class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = {}  # Tabela Q que o agente usará para armazenar os valores de aprendizado
        self.actions = actions  # Lista de ações possíveis que o agente pode tomar
        self.alpha = alpha  # Taxa de aprendizado (o quanto o agente atualiza valores antigos)
        self.gamma = gamma  # Fator de desconto (o quanto o agente valoriza recompensas futuras)
        self.epsilon = epsilon  # Probabilidade de explorar (tentar ações aleatórias)

    def get_q_value(self, state, action):
        # Retorna o valor Q para um estado e ação específicos (ou 0.0 se ainda não foi definido)
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state):
        # Escolhe uma ação com base na estratégia epsilon-greedy
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)  # Exploração: tenta uma ação aleatória
        else:
            # Exploração: escolhe a melhor ação com base na tabela Q atual
            q_values = {action: self.get_q_value(state, action) for action in self.actions}
            return max(q_values, key=q_values.get)

    def update_q_value(self, state, action, reward, next_state):
        # Atualiza o valor Q usando a equação do Q-Learning
        current_q = self.get_q_value(state, action)  # Valor Q atual
        max_next_q = max([self.get_q_value(next_state, a) for a in self.actions])  # Melhor Q do próximo estado
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)  # Atualização
        self.q_table[(state, action)] = new_q  # Armazena o novo valor Q na tabela

# Inicializa o ambiente e o agente
environment = SmartHomeEnvironment()
actions = ["ligar_AC", "desligar_AC", "ligar_Luzes", "desligar_Luzes"]
agent = QLearningAgent(actions)

# Listas para monitorar o desempenho
episodes = 1000  # Número de episódios de treinamento
rewards_per_episode = []  # Recompensas totais por episódio
epsilon_values = []  # Valores de epsilon ao longo do tempo

# Ciclo de treinamento
for episode in range(episodes):
    state = environment.get_state()  # Obtém o estado inicial do ambiente
    total_reward = 0  # Soma da recompensa para este episódio
    
    for step in range(10):  # Interações dentro de um episódio
        action = agent.choose_action(state)  # Escolhe uma ação
        reward = environment.get_reward(action)  # Obtém a recompensa para a ação escolhida
        total_reward += reward  # Soma a recompensa total do episódio
        environment.update_environment()  # Atualiza o ambiente para o próximo passo
        next_state = environment.get_state()  # Obtém o próximo estado
        agent.update_q_value(state, action, reward, next_state)  # Atualiza a tabela Q
        state = next_state  # Avança para o próximo estado
    
    rewards_per_episode.append(total_reward)  # Salva a recompensa total do episódio
    epsilon_values.append(agent.epsilon)  # Salva o valor de epsilon
    agent.epsilon = max(0.01, agent.epsilon * 0.995)  # Decaimento do epsilon para menos exploração

# Teste do agente após o treinamento
environment = SmartHomeEnvironment()
for step in range(10):
    state = environment.get_state()
    action = agent.choose_action(state)
    print(f"Estado: {state}, Ação Escolhida: {action}")
    environment.update_environment()

# Gera o gráfico de recompensas por episódio
plt.figure(figsize=(12, 6))
plt.plot(rewards_per_episode, label="Recompensa por Episódio")
plt.xlabel("Episódios")
plt.ylabel("Recompensa Total")
plt.title("Desempenho do Agente ao Longo do Treinamento")
plt.legend()
plt.grid()
plt.show()

# Gera o gráfico de decaimento do epsilon
plt.figure(figsize=(12, 6))
plt.plot(epsilon_values, label="Valor de Epsilon")
plt.xlabel("Episódios")
plt.ylabel("Epsilon")
plt.title("Decaimento de Epsilon ao Longo do Treinamento")
plt.legend()
plt.grid()
plt.show()
