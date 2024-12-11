import torch

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    使用Torch张量并在GPU上计算广义优势估计(GAE)
    要求rewards, values, dones已在GPU或指定device上。
    
    Args:
        rewards: torch.Tensor, shape (T,), float32
        values: torch.Tensor, shape (T+1,), float32
        dones:   torch.Tensor, shape (T,), float32 (0或1表示是否结束)
        gamma: 折扣因子
        lam: GAE因子
    
    Returns:
        advantages: torch.Tensor, shape(T,)
        returns:    torch.Tensor, shape(T,)
    """
    T = rewards.shape[0]
    advantages = torch.zeros(T, dtype=torch.float32, device=rewards.device)
    last_adv = 0.0

    # 使用反向迭代计算GAE
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
        last_adv = delta + gamma * lam * (1 - dones[t]) * last_adv
        advantages[t] = last_adv
    values = values.squeeze(dim=1).squeeze(dim=1)
    returns = values[1:] + advantages
    return advantages, returns