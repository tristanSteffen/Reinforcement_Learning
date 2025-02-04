import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class DQNNetwork(nn.Module):
    def __init__(self, state_size=9, action_size=9, hidden_size=64):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        # x shape: [batch_size, state_size]
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # returns Q-values for each action (shape [batch_size, 9])

class DQNAgent:
    """
    DQN Agent for Tic-Tac-Toe.
    State: a 1D array of size 9.
    Actions: 0..8 (each board position).
    """

    def __init__(
        self,
        state_size=9,
        action_size=9,
        lr=1e-4,  # Learning rate
        gamma=0.8,  # Discount factor
        epsilon=1,  # Exploration rate
        batch_size=512,  # Batch size for replay
        max_memory=50000  # Max size of replay memory
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma  # Store gamma as an instance attribute
        self.epsilon = epsilon  # Store epsilon as an instance attribute
        self.batch_size = batch_size

        # Store learning rate as an instance attribute
        self.lr = lr

        # Replay memory
        self.memory = deque(maxlen=max_memory)

        # Two networks: policy and target
        self.policy_net = DQNNetwork()
        self.target_net = DQNNetwork()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Initialize optimizer with the provided learning rate
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

    def act(self, state, valid_actions=None):
        """
        Epsilon-greedy action selection.
        state: np.array of shape (9,) for Tic-Tac-Toe
        valid_actions: a list of valid actions (some squares are taken).
        """
        # With probability epsilon, pick random valid action
        if np.random.rand() < self.epsilon:
            if valid_actions:
                return random.choice(valid_actions)
            else:
                return random.randint(0, self.action_size - 1)
        else:
            # Otherwise pick the best action from Q-network
            state_t = torch.FloatTensor(state).unsqueeze(0)  # shape [1,9]
            with torch.no_grad():
                q_values = self.policy_net(state_t)[0]  # shape [9]
            q_values = q_values.cpu().numpy()

            # Mask out invalid actions by setting them to -inf
            if valid_actions is not None:
                masked_q = np.full(self.action_size, -np.inf)
                for a in valid_actions:
                    masked_q[a] = q_values[a]
                action = int(np.argmax(masked_q))
            else:
                action = int(np.argmax(q_values))
            return action
        
    def remember(self, state, action, reward, next_state, done):
        """Store the transition in replay memory."""
        self.memory.append((state, action, reward, next_state, done))
        
    def replay(self, batch_size=None):
        """Trainiert das DQN mit Experience Replay und der Bellman-Formel."""
        if batch_size is None:
            batch_size = self.batch_size

        if len(self.memory) < batch_size:
            return  # Nicht genug gespeicherte Erlebnisse, daher kein Training

        # Zufällige Stichprobe von Erfahrungen aus dem Replay Buffer
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Umwandlung der Listen in Tensoren für effizientere Berechnungen
        states_t = torch.FloatTensor(np.array(states))
        next_states_t = torch.FloatTensor(np.array(next_states))
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1)  # Belohnung als Spaltenvektor
        actions_t = torch.LongTensor(actions).unsqueeze(1)   # Aktionen als Spaltenvektor
        dones_t = torch.FloatTensor(dones).unsqueeze(1)      # Boolean-Werte für Endzustände

        # 1️⃣ **Berechnung von Q(s, a)**
        # Hole Q-Werte für die gewählten Aktionen
        current_q = self.policy_net(states_t).gather(1, actions_t)

        # 2️⃣ **Anwendung der Bellman-Gleichung**
        with torch.no_grad():  # Deaktiviert Gradientenberechnung, da wir nur Werte berechnen
            next_q_values = self.target_net(next_states_t)
            min_next_q = next_q_values.min(1, keepdim=True)[0]  # Opponent minimizes agent's value
            target_q = rewards_t + (1 - dones_t) * self.gamma * min_next_q

        # 3️⃣ **Berechnung des Verlusts & Optimierung**
        loss = self.loss_fn(current_q, target_q)  # MSE-Loss zwischen aktuellem und Ziel-Q-Wert
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)  # Verhindert zu große Updates
        self.optimizer.step()  # Parameter-Update

    def update_target_network(self):
        """Copy policy_net weights to target_net."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_learning_rate(self, new_lr):
        """Update the learning rate of the optimizer."""
        self.lr = new_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
            
    def update_epsilon(self, new_epsilon):
        """Update the epsilon of the optimizer."""
        self.epsilon = new_epsilon
        for param_group in self.optimizer.param_groups:
            param_group['epsilon'] = self.epsilon

    def save(self, filename="dqn_tictactoe.pt"):
        torch.save(self.policy_net.state_dict(), filename)

    def load(self, filename="dqn_tictactoe.pt"):
        self.policy_net.load_state_dict(
            torch.load(filename, weights_only=True)
        )
        self.update_target_network()