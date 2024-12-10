import torch
import numpy as np
from env import RedirectWalkingEnv
from model import PolicyValueNet
from ppo import PPO
from utils import compute_gae
import json

def main():
    # 初始化环境和网络
    config_path = "config.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        env_config = json.load(f)
    env = RedirectWalkingEnv(env_config)
    state_dim = env.observation_space.shape[0]
    action_dim = int(env.action_space.shape[0] / 2)
    
    net = PolicyValueNet(state_dim, action_dim)
    agent = PPO(net, lr=3e-4, gamma=0.99, lam=0.95, clip_range=0.1, ent_coef=0.001)
    
    max_episodes = 300
    horizon = 2048  # 每次收集的数据长度
    episode_rewards = []
    
    for ep in range(max_episodes):
        # Rollout buffer
        states = []
        actions = []
        raw_actions = []
        rewards = []
        dones = []
        values = []
        log_probs = []
        
        state = env.reset()
        
        for t in range(horizon):
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action, log_prob, value, raw_action = agent.get_action(state_t)
            action = action.squeeze(0).numpy()
            raw_action = raw_action.squeeze(0).numpy()
            
            next_state, reward, done, info = env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            raw_actions.append(raw_action)
            dones.append(float(done))
            values.append(value.item())
            log_probs.append(log_prob.item())
            
            state = next_state

        
        # 收集最后一状态的价值
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            _, _, last_value = net(state_t)
        values.append(last_value.item())
        
        advantages, returns = compute_gae(np.array(rewards), np.array(values), np.array(dones),
                                          gamma=agent.gamma, lam=agent.lam)
        
        # 对优势进行标准化
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        rollout = {
            'states': states,
            'log_probs': log_probs,
            'returns': returns,
            'advantages': advantages,
            'raw_actions': raw_actions
        }

        
        # PPO 更新
        agent.update(rollout)
        
        # 简单的日志输出，可根据需要进行扩展
        episode_reward = np.sum(rewards)
        episode_rewards.append(episode_reward)
        print(f"Episode {ep} finished. Total Reward: {episode_reward}")
        
    # draw
    import matplotlib.pyplot as plt
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Curve')
    plt.show()
    

if __name__ == "__main__":
    main()
