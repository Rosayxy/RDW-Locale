import numpy as np

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    计算广义优势估计 (GAE)
    Args:
        rewards: shape (T,)
        values: shape (T+1,)
        dones: shape (T,)
    Returns:
        advantages, returns
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_adv = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
        advantages[t] = last_adv = delta + gamma * lam * (1 - dones[t]) * last_adv
    returns = values[:-1] + advantages
    return advantages, returns
