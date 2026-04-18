import torch
import torch.nn as nn
import torch.optim as optim

class PPO(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PPO, self).__init__()
        # Actor: 输出动作的概率分布
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        # Critic: 预测状态的价值 (V值)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def update(self, states, actions, old_log_probs, returns, advantages):
        # 1. 计算当前策略的 log 概率
        curr_probs = self.actor(states)
        dist = torch.distributions.Categorical(curr_probs)
        curr_log_probs = dist.log_prob(actions)
        
        # 2. 计算新旧策略的比率 r(theta)
        ratio = torch.exp(curr_log_probs - old_log_probs)

        # 3. PPO 核心截断逻辑
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1-0.2, 1+0.2) * advantages
        
        # Actor 损失：取最小值并取负（因为要最大化奖励）
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # Critic 损失：均方误差
        critic_loss = nn.MSELoss()(self.critic(states), returns)
        
        # 反向传播更新
        loss = actor_loss + 0.5 * critic_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
