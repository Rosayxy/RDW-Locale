import torch
import numpy as np
from env import RedirectWalkingEnv
from model import PolicyValueNet
from ppo import PPO
from utils import compute_gae
import json    
import matplotlib.pyplot as plt
import time

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    # 初始化环境和网络
    config_path = "config.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        env_config = json.load(f)
    start_time = time.time()
    env = RedirectWalkingEnv(env_config)
    print("Env init time:", time.time() - start_time)
    start_time = time.time()
    state_dim = env.observation_space.shape[0]
    action_dim = int(env.action_space.shape[0] / 2)
    
    net = PolicyValueNet(state_dim, action_dim)
    agent = PPO(net, lr=3e-4, gamma=0.99, lam=0.95, clip_range=0.1, ent_coef=0.001,device=device)
    
    max_episodes = 10
    episode_rewards = []
    print("Net init time:", time.time() - start_time)
    for ep in range(max_episodes):
        start_time = time.time()
        # Rollout buffer
        states = []
        raw_actions = []
        rewards = []
        dones = []
        values = []
        log_probs = []
        
        state = env.reset()
        print("Env reset time:", time.time() - start_time)
        start_time = time.time()
        done = False
        while not done:
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action, log_prob, value, raw_action = agent.get_action(state_t)
            action = action.squeeze(0).numpy()
            
            next_state, reward, done, info = env.step(action)
            
            states.append(state)
            rewards.append(reward)
            raw_actions.append(raw_action)
            dones.append(float(done))
            values.append(value)
            log_probs.append(log_prob)
            
            state = next_state
        print("Env step time:", time.time() - start_time)
        start_time = time.time()
        # 收集最后一状态的价值
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            _, _, last_value = net(state_t)
        values.append(last_value)
        
        # 格式转换
        states = states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        log_probs = torch.stack(log_probs).to(device)
        values = torch.stack(values).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)
        raw_actions = torch.stack(raw_actions).to(device)
        
        
        advantages, returns = compute_gae(rewards, values, dones,
                                          gamma=agent.gamma, lam=agent.lam)
        print("Compute GAE time:", time.time() - start_time)
        start_time = time.time()
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
        print("PPO update time:", time.time() - start_time)
        # 简单的日志输出，可根据需要进行扩展
        episode_reward = rewards.sum().item()
        episode_rewards.append(episode_reward)
        print(f"Episode {ep} finished. Total Reward: {episode_reward}")
        
        # 每100个episode保存一次模型，文件名包括eposide reward
        if (ep+1) % 100 == 0:
            torch.save(net.state_dict(), f'ppo_net_{ep}_{episode_reward}.pth')
            plt.plot(episode_rewards)
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.title('Training Curve')
            plt.savefig(f'training_curve_{ep}_{episode_reward}.png')
            plt.close()
        
    # draw

    
    

if __name__ == "__main__":
    main()
