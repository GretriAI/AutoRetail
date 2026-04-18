import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

# 1. 定义 Q 网络 (一个简单的全连接网络)
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.fc(x)

# 2. DQN 智能体核心逻辑
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.q_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict()) # 初始权重一致
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=0.001)
        self.memory = [] # 经验回放池
        self.gamma = 0.99

    def train(self, batch_size=32):
        if len(self.memory) < batch_size: return
        
        # 从经验池随机采样 (s, a, r, s', done)
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # 转换为 Tensor
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

        # 计算当前的 Q 值
        current_q = self.q_net(states).gather(1, actions)

        # 计算目标 Q 值 (使用 Target Network)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        # 计算 Loss 并更新主网络
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
