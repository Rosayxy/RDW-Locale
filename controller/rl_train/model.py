import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyValueNet(nn.Module):
    def __init__(self, state_dim, action_dim=3):
        super(PolicyValueNet, self).__init__()
        # 论文中描述：两子网络共享前两层参数
        # 全连接层维度：每层128个神经元，激活为Tanh
        
        # 共享特征提取层
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        
        # 策略头(输出为动作的均值和标准差)
        # mean与log_std可以分别输出，也可以将log_std作为参数独立训练
        self.policy_mean = nn.Linear(128, action_dim) 
        self.policy_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # 值函数头
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        # x: (batch, state_dim)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        
        # 策略输出
        mean = self.policy_mean(x)
        log_std = self.policy_log_std.expand_as(mean)
        
        # 值输出
        value = self.value_head(x)
        return mean, log_std, value
