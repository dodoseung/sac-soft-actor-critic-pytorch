# Haarnoja, Tuomas, et al. "Soft actor-critic algorithms and applications." arXiv preprint arXiv:1812.05905 (2018).
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

import numpy as np
import gym
import random
from collections import deque

class ReplayBuffer():
    def __init__(self, max_size=100000):
        super(ReplayBuffer, self).__init__()
        self.max_size = max_size
        self.memory = deque(maxlen=self.max_size)
        
    # Add the replay memory
    def add(self, state, action, reward, next_state, done, last_action, hidden_in, hidden_out):
        self.memory.append((state, action, reward, next_state, done, last_action, hidden_in, hidden_out))

    # Sample the replay memory
    def sample(self, batch_size):
        batch = random.sample(self.memory, min(batch_size, len(self.memory)))
        states, actions, rewards, next_states, dones, last_action, hidden_in, hidden_out = map(np.stack, zip(*batch))
        return states, actions, rewards, next_states, dones, last_action, hidden_in, hidden_out

class SoftQNet(nn.Module):
    def __init__(self, state_num, action_num):
        super(SoftQNet, self).__init__()
        # Network 1
        self.input1 = nn.Linear(state_num + action_num, 512)
        self.fc1 = nn.Linear(state_num + action_num, 512)
        self.lstm1 = nn.LSTM(512, 512)
        self.fc2 = nn.Linear(2 * 512, 512)
        self.output1 = nn.Linear(512, 1)
        
        # Network 2
        self.input2 = nn.Linear(state_num + action_num, 512)
        self.fc3 = nn.Linear(state_num + action_num, 512)
        self.lstm2 = nn.LSTM(512, 512)
        self.fc4 = nn.Linear(2 * 512, 512)
        self.output2 = nn.Linear(512, 1)
        
    def forward(self, x, u, last_u, hidden):
        # x = x.permute(1,0,2)
        # u = u.permute(1,0,2)
        # last_u = last_u.permute(1,0,2)
        
        # Network 1
        # FC1
        x1 = torch.cat([x, u], 1) 
        x1 = F.relu(self.input1(x1))
        # LSTM
        x2 = torch.cat([x, last_u], 1)
        x2 = F.relu(self.fc1(x2))
        x2, _ = self.lstm1(x2, hidden)
        # FC2
        x3 = torch.cat([x1, x2], 1) 
        x3 = F.relu(self.fc2(x3))
        value1 = F.relu(self.output1(x3))
        value1 = value1.permute(1,0,2)
        
        # Network 2
        # FC1
        x1 = torch.cat([x, u], 1) 
        x1 = F.relu(self.input2(x1))
        # LSTM
        x2 = torch.cat([x, last_u], 1)
        x2 = F.relu(self.fc3(x2))
        x2, _ = self.lstm2(x2, hidden)
        # FC2
        x3 = torch.cat([x1, x2], 1) 
        x3 = F.relu(self.fc4(x3))
        value2 = F.relu(self.output2(x3))
        value2 = value2.permute(1,0,2)
        
        return value1, value2

class PolicyNet(nn.Module):
    def __init__(self, state_num, action_num, min_action, max_action):
        super(PolicyNet, self).__init__()
        self.min_action = min_action
        self.max_action = max_action
        
        self.input = nn.Linear(state_num, 512)
        self.fc1 = nn.Linear(state_num + action_num, 512)
        self.lstm1 = nn.LSTM(512, 512)
        self.fc2 = nn.Linear(2 * 512, 512)
        self.output = nn.Linear(512, 512)
        
        self.mu = nn.Linear(512, 1)
        self.std = nn.Linear(512, 1)
        
    def forward(self, x, last_u, hidden):
        # x = x.permute(1,0,2)
        # last_u = last_u.permute(1,0,2)
        
        # FC1
        x1 = F.relu(self.input(x))
        
        # LSTM
        x2 = torch.cat([x, last_u], 1)
        x2 = F.relu(self.fc1(x2))
        x2, hidden = self.lstm1(x2, hidden)
        
        # FC2
        x3 = torch.cat([x1, x2], 1) 
        x3 = F.relu(self.fc2(x3))
        x3 = F.relu(self.output(x3))
        x3 = x3.permute(1,0,2)
        
        # Mean and stdev
        mu = (self.max_action - self.min_action) * F.sigmoid(self.mu(x)) + self.min_action
        std = (self.max_action - self.min_action) * F.sigmoid(self.std(x)) / 2

        return mu, std, hidden
    
    # Enforcing action bounds
    def sample(self, states, last_u, hidden, epsilon=1e-6):
        mu, std = self.forward(states, last_u, hidden)
        dist = D.Normal(mu, std)
        actions = dist.rsample()
        log_probs = dist.log_prob(actions) - torch.log(1. - torch.tanh(actions).pow(2) + epsilon)
        return actions, log_probs
    
class SAC():
    def __init__(self, env, memory_size=1000000, batch_size=64, gamma=0.95, learning_rate=1e-3, tau=0.01, target_entropy=-1, reward_normalization=False, reward_scale=5):
        super(SAC, self).__init__()
        self.env = env
        self.state_num = self.env.observation_space.shape[0]
        self.action_num = self.env.action_space.shape[0]
        self.action_max = float(env.action_space.high[0])
        self.action_min = float(env.action_space.low[0])
                
        # Torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Soft Q
        self.soft_q_net = SoftQNet(self.state_num, self.action_num).to(self.device)
        self.soft_q_opt = optim.Adam(self.soft_q_net.parameters(), lr=learning_rate)
        
        # Soft Q target
        self.soft_q_target_net = SoftQNet(self.state_num, self.action_num).to(self.device)
        self.soft_q_target_net.load_state_dict(self.soft_q_net.state_dict())
        
        # Policy
        self.policy_net = PolicyNet(self.state_num, self.action_num, self.action_min, self.action_max).to(self.device)
        self.policy_opt = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Temperature parameter alpha
        self.alpha = torch.ones(1, dtype=torch.float32, requires_grad=True, device=self.device)
        self.alpha_opt = optim.Adam([self.alpha], lr=learning_rate)
        self.target_entropy = target_entropy
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(memory_size)
        self.batch_size = batch_size
        
        # Learning setting
        self.gamma = gamma
        self.tau = tau
        
        # Reward setting
        self.reward_normalization = reward_normalization
        self.reward_scale = reward_scale
        
    # Get the action
    def get_action(self, state, last_action, hidden_in):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        last_action = torch.FloatTensor(last_action).to(self.device).unsqueeze(0)
        mu, std, hidden_out = self.policy_net(state, last_action, hidden_in)
        action = D.Normal(mu, std).sample()
        action = action.cpu().detach().numpy()
        return action[0], hidden_out

    # Soft update a target network
    def soft_update(self, net, target_net):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def learn(self):
        # Get memory from replay buffer
        states, actions, rewards, next_states, dones, last_action, hidden_in, hidden_out = self.replay_buffer.sample(self.batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device).view(-1, 1)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device).view(-1, 1)
        last_actions = torch.FloatTensor(last_action).to(self.device)
        hidden_in = torch.FloatTensor(hidden_in).to(self.device)
        hidden_out = torch.FloatTensor(hidden_out).to(self.device)
 
        # Reward normalization and scailing
        rewards = self.reward_scale * (rewards - rewards.mean()) / rewards.std() if self.reward_normalization else rewards
        
        # Get the new next action with the current policy
        new_next_actions, next_log_probs = self.policy_net.sample(next_states, actions, hidden_out)

        # Get the target q value
        next_q_value_1, next_q_value_2 = self.soft_q_target_net(next_states, new_next_actions, actions, hidden_out)
        next_q_values = torch.min(next_q_value_1, next_q_value_2)
        target_q_values = rewards + self.gamma * (next_q_values - self.alpha * next_log_probs) * (1-dones)

        # Calculate the q value loss and optimize the q value network
        q_value_1, q_value_2 = self.soft_q_net(states, actions, last_actions, hidden_in)
        q_loss =  F.mse_loss(q_value_1, target_q_values.detach()) + F.mse_loss(q_value_2, target_q_values.detach())
        self.soft_q_opt.zero_grad()
        q_loss.backward()
        self.soft_q_opt.step()
        
        # Get the new action with the current policy
        new_actions, log_probs = self.policy_net.sample(states, last_actions, hidden_in)
        
        # Get the minimum q value
        new_q_value_1, new_q_value_2 = self.soft_q_net(states, new_actions, last_actions, hidden_in)
        q_values = torch.min(new_q_value_1, new_q_value_2)

        # Calculate the policy loss and optimize the policy network
        policy_loss = (self.alpha * log_probs - q_values).mean()
        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()
        
        # Automating Entropy Adjustment for Maximum Entropy 
        alpha_loss = (-self.alpha * log_probs.detach() - self.alpha * self.target_entropy).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        
        # Soft update the value network
        self.soft_update(self.soft_q_net, self.soft_q_target_net)

        
def main():
    env = gym.make("Pendulum-v0")
    agent = SAC(env, memory_size=1000000, batch_size=128, gamma=0.99, learning_rate=3e-4, tau=0.01, target_entropy=-1, reward_normalization=False, reward_scale=10)
    ep_rewards = deque(maxlen=1)
    total_episode = 10000

    for i in range(total_episode):
        state = env.reset()
        last_action = env.action_space.sample()
        hidden_out = (torch.zeros([1, 1, 512], dtype=torch.float32, device=agent.device), torch.zeros([1, 1, 512], dtype=torch.float32, device=agent.device))
        hidden_in = hidden_out
        
        ep_reward = 0
        while True:
            action, hidden_out = agent.get_action(state, last_action, hidden_in)
            next_state, reward , done, _ = env.step(action)
            ep_reward += reward

            agent.replay_buffer.add(state, action, reward, next_state, done, last_action, hidden_in, hidden_out)
            if i > 2:
                agent.learn()
            
            if done:
                ep_rewards.append(ep_reward)
                if i % 1 == 0:
                    print("episode: {}\treward: {}".format(i, round(np.mean(ep_rewards), 3)))
                break

            state = next_state
            last_action = action
            hidden_in = hidden_out

if __name__ == '__main__':
    main()
        