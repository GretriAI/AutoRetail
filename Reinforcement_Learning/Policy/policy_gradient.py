import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import gymnasium as gym

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, action_dim)

    def forward(self, x):
        # pi_theta: maps state to action probabilities
        return F.softmax(self.fc2(F.relu(self.fc1(x))), dim=-1)

class REINFORCEAgent:
    def __init__(self, state_dim, action_dim):
        # FORMULA: Initialize pi_theta to anything
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-2)
        self.gamma = 0.99
        self.memory = []

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        
        # FORMULA: log pi_theta(an | sn)
        # We store the log probability to use in the update step later
        return action.item(), m.log_prob(action)

    def update(self):
        running_g = 0
        returns = []

        # FORMULA: Gn <- sum_{t=0}^{T-n} gamma^t * r_{n+t}
        # We calculate the cumulative discounted return Gn for every step n
        for _, reward in reversed(self.memory):
            running_g = reward + self.gamma * running_g
            returns.insert(0, running_g)
        
        returns = torch.tensor(returns)
        
        # FORMULA: Update policy theta <- theta + alpha * gamma^n * Gn * grad(log pi)
        loss = []
        for i, (log_prob, _) in enumerate(self.memory):
            # Note: In practice, we omit gamma^n as it stabilizes training
            # "loss" is negative because optimizers minimize, but we want to maximize J(pi)
            g_n = returns[i]
            loss.append(-log_prob * g_n)

        # Apply the update (backpropagation handles the Gradient "grad" part)
        self.optimizer.zero_grad()
        total_loss = torch.stack(loss).sum()
        total_loss.backward()  # This computes the "grad" (nabla) of the objective
        self.optimizer.step()  # This applies the "+ alpha" update to theta
        
        self.memory = []

# --- Execution Loop ---

# FORMULA: Loop forever (for each episode)
env = gym.make('CartPole-v1')
agent = REINFORCEAgent(4, 2)

for episode in range(500):
    state, _ = env.reset()
    
    # FORMULA: Generate episode s0, a0, r0 ... sT, aT, rT
    for t in range(1000):
        action, log_prob = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        
        agent.memory.append((log_prob, reward))
        state = next_state
        if terminated or truncated: break
            
    # FORMULA: Return pi_theta (Implicitly update the policy weights)
    agent.update()
