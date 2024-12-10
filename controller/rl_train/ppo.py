import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.nn.functional import mse_loss
import numpy as np


DELTA_T=0.02 # 50 fps
MIN_TRANS_GAIN=0.86
MAX_TRANS_GAIN=1.26
MIN_ROT_GAIN=0.67
MAX_ROT_GAIN=1.24
MIN_CUR_GAIN_R=750 # cm!
INF_CUR_GAIN_R=2500000 # cm!

class PPO:
    def __init__(self, policy_value_net, lr=3e-4, gamma=0.99, lam=0.95, clip_range=0.1, ent_coef=0.001):
        self.net = policy_value_net
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        
        self.gamma = gamma
        self.lam = lam
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        
    def get_action(self, state):
        with torch.no_grad():
            mean, log_std, value = self.net(state)
            std = torch.exp(log_std)
            dist = Normal(mean, std)
            
            raw_action = dist.sample()  # 原始高斯样本
            log_prob = dist.log_prob(raw_action).sum(axis=-1)
        
        # 对raw_action进行tanh映射[-1,1]再映射到实际范围
        action = raw_action.clone()
        # 将[-1,1]范围映射到指定区间 [min, max]: action = (tanh(x)+1)/2 * (max-min) + min
        
        # 对第0维(平移增益)
        action[...,0] = ((torch.tanh(raw_action[...,0]) + 1)/2)*(MAX_TRANS_GAIN - MIN_TRANS_GAIN) + MIN_TRANS_GAIN
        # 对第1维(旋转增益)
        action[...,1] = ((torch.tanh(raw_action[...,1]) + 1)/2)*(MAX_ROT_GAIN - MIN_ROT_GAIN) + MIN_ROT_GAIN
        # 对第2维(曲率增益)
        action[...,2] = ((torch.tanh(raw_action[...,2]) + 1)/2)*(INF_CUR_GAIN_R - MIN_CUR_GAIN_R) + MIN_CUR_GAIN_R


        return action, log_prob, value, raw_action
    
    def evaluate_actions(self, states, raw_actions):
        mean, log_std, values = self.net(states)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        
        log_probs = dist.log_prob(raw_actions).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        
        return log_probs, values, entropy
    
    def update(self, rollout):
        # rollout 包含：states, actions, old_log_probs, returns, advantages
        
        states = torch.from_numpy(np.array(rollout['states'], dtype=np.float32))
        raw_actions = torch.from_numpy(np.array(rollout['raw_actions'], dtype=np.float32))
        old_log_probs = torch.tensor(rollout['log_probs'], dtype=torch.float32)
        returns = torch.tensor(rollout['returns'], dtype=torch.float32)
        advantages = torch.tensor(rollout['advantages'], dtype=torch.float32)
        
        # PPO更新，一般会进行多次优化迭代
        # 假设进行K次更新
        K = 10
        batch_size = len(states)
        for _ in range(K):
            log_probs, values, entropy = self.evaluate_actions(states,  raw_actions)
            
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (returns - values).pow(2).mean()
            entropy_loss = -entropy.mean()
            
            loss = actor_loss + 0.5 * critic_loss + self.ent_coef * entropy_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
