import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class SimpleGRPO:
    def __init__(self, policy_model, optimizer, beta=0.1, epsilon=0.2):
        self.policy = policy_model  # 策略网络 (Actor)
        self.optimizer = optimizer
        self.beta = beta            # KL 散度惩罚系数
        self.epsilon = epsilon      # PPO Clip 的超参数

    def compute_grpo_loss(self, prompts, old_policy):
        """
        核心 GRPO 损失函数
        prompts: 输入的提示词 batch
        old_policy: 用于采样的旧策略网络（冻结梯度）
        """
        # 设定组大小 G (Group Size)
        # GRPO 的核心：对同一个 prompt 采样个不同的回答
        group_size = 4 
        
        all_losses = []
        
        for prompt in prompts:
            # 1. 使用旧策略网络采样 G 个回答 (Outputs)
            with torch.no_grad():
                # 这里的 sample_responses 是一个虚拟函数，表示针对一个 prompt 采样 G 个序列
                # 实际大模型中会包含 inputs_ids, attention_mask 等
                responses, old_log_probs = old_policy.sample_responses(prompt, num_samples=group_size)
            
            # 2. 模拟奖励模型（Reward Model）对这 G 个回答打分
            # 在实际应用中，这里会调用一个 Reward 模型或者规则（如编译器、数学正确性检查）
            rewards = self.mock_reward_model(prompt, responses) # 形状: [group_size]
            
            # 3. 【GRPO 的精髓】计算组内相对奖励 (Relative Rewards)
            # 对当前 Prompt 的 G 个奖励进行标准化（减去均值，除以标准差）
            mean_r = rewards.mean()
            std_r = rewards.std() + 1e-8
            advantages = (rewards - mean_r) / std_r  # 形状: [group_size]
            
            # 4. 用当前正在训练的策略网络计算新的 log probabilities
            # 这里的 forward_responses 也是虚拟函数，计算当前模型生成这些回答的概率
            new_log_probs, ref_log_probs = self.policy.forward_responses(prompt, responses)
            
            # 5. 计算重要性采样比率 (Probability Ratio)
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # 6. 计算 PPO Clip 损失
            surr1 = ratio * advantages.unsqueeze(-1) # 广播机制
            surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantages.unsqueeze(-1)
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 7. 计算 KL 散度约束（防止当前策略偏离参考模型/旧模型太远）
            # KL 分割公式: kl = exp(ref_log_prob - new_log_prob) - (ref_log_prob - new_log_prob) - 1
            kl = torch.exp(ref_log_probs - new_log_probs) - (ref_log_probs - new_log_probs) - 1
            kl_loss = self.beta * kl.mean()
            
            # 总损失
            total_loss = policy_loss + kl_loss
            all_losses.append(total_loss)
            
        # 整个 Batch 的平均损失
        return torch.stack(all_losses).mean()

    def train_step(self, prompts, old_policy):
        self.optimizer.zero_grad()
        loss = self.compute_grpo_loss(prompts, old_policy)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def mock_reward_model(self, prompt, responses):
        """ 这是一个模拟的奖励函数 """
        # 在大模型落地时（如 DeepSeek-R1-Zero），这里往往是：
        # 是否包含思考标签 <think></think> + 答案是否正确（LeetCode/数学题等）
        return torch.randn(len(responses)) # 随机返回 G 个得分