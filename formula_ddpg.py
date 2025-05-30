
import os
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from itertools import count
from formula_env import FormulaEnv
from gym.envs.registration import register

register(id='FormulaEnv-v0', entry_point='formula_env:FormulaEnv')

# Hyper-parameters
mode = 'train'  # 'train' or 'test'
env_name = "FormulaEnv-v0"  # Ensure this is a valid Gym environment ID
tau = 0.005
target_update_interval = 1
test_iteration = 10

learning_rate = 1e-7
gamma = 0.99
buffer_capacity = 1000000
batch_size = 100
random_seed = 9527

# Optional parameters
sample_frequency = 2000
render = True
log_interval = 50
load = False
render_interval = 100
exploration_noise = 0.05
max_episode = 1000
print_log = 5
update_iteration = 200
max_length_of_trajectory = 100

# Global Variables
device = 'cuda' if torch.cuda.is_available() else 'cpu'
script_name = os.path.basename(__file__)
env = FormulaEnv(render_mode='human')
directory = './exp_ddpg_' + env_name 

state_dim = env.state_dim
action_dim = env.action_dim
max_action = [float(env.action_space["acceleration"].high[0]), float(env.action_space["steering_velocity"].high[0])]
min_action = [float(env.action_space["acceleration"].low[0]), float(env.action_space["steering_velocity"].low[0])]
min_Val = torch.tensor(1e-7).float().to(device) # min value

# Set random seed
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if device == 'cuda':
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ReplayBuffer():
    '''
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects tuples of (state, next_state, action, reward, done)
    '''
    def __init__(self, max_size=buffer_capacity):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)

    
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 200)
        self.l4 = nn.Linear(200, 200)
        self.l5 = nn.Linear(200, 100)
        self.l6 = nn.Linear(100, action_dim)

        self.register_buffer("max_action", torch.tensor(max_action, dtype=torch.float32))

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = F.relu(self.l5(x))
        x = self.max_action * torch.tanh(self.l6(x))
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400 , 300)
        self.l3 = nn.Linear(300, 200)
        self.l4 = nn.Linear(200, 200)
        self.l5 = nn.Linear(200, 100)
        self.l6 = nn.Linear(100, 1)

    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat([x, u ], -1)))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = F.relu(self.l5(x))
        x = self.l6(x)
        return x


class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.writer = SummaryWriter(directory)

        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0
        
        self.state_mean = torch.zeros(state_dim).to(device)
        self.state_std = torch.ones(state_dim).to(device)
        
    def compute_state_stats(self):
        # Compute mean and std of the states in the replay buffer
        states = np.array([data[0] for data in self.replay_buffer.storage])
        states = np.array(states)
        self.state_mean = torch.mean(torch.tensor(states, dtype=torch.float32), dim=0).to(device)
        self.state_std = torch.std(torch.tensor(states, dtype=torch.float32), dim=0).to(device)
        
    def normalize_state(self, state):
        # Normalize the state using the computed mean and std
        state = torch.tensor(state, dtype=torch.float32).to(device)
        state = (state - self.state_mean) / (self.state_std)
        return state
        
    def select_action(self, state):
        state = self.normalize_state(state).cpu().numpy()
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self):
        if len(self.replay_buffer.storage) < batch_size:
            return
        
        self.compute_state_stats()
        for it in range(update_iteration):
            # Sample replay buffer
            x, y, u, r, d = self.replay_buffer.sample(batch_size)
            state = self.normalize_state(x)
            action = torch.FloatTensor(u).to(device)
            next_state = self.normalize_state(y)
            done = torch.FloatTensor(1-d).to(device)
            reward = torch.FloatTensor(r).to(device)

            # Compute the target Q value
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + ((1-done) * gamma * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)
            self.writer.add_scalar('Loss/critic_loss', critic_loss, global_step=self.num_critic_update_iteration)
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()
            self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1

    def save(self):
        torch.save(self.actor.state_dict(), directory + 'actor.pth')
        torch.save(self.critic.state_dict(), directory + 'critic.pth')
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def load(self):
        self.actor.load_state_dict(torch.load(directory + 'actor.pth'))
        self.critic.load_state_dict(torch.load(directory + 'critic.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")

#TODO

def main():
    agent = DDPG(state_dim, action_dim, max_action)
    ep_r = 0
    total_reward_array = []
    if mode == 'test':
        agent.load()
        for i in range(test_iteration):
            state = env.reset()
            for t in count():
                action = agent.select_action(state)
                next_state, reward, done, _,  info = env.step(np.float32(action))
                ep_r += reward
                env.render()
                if done or t >= max_length_of_trajectory:
                    print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(i, ep_r, t))
                    ep_r = 0
                    break
                state = next_state

    elif mode == 'train':
        if load: agent.load()
        total_step = 0
        for i in range(max_episode):
            total_reward = 0
            step =0
            state = env.reset()
            for t in count():
                action = agent.select_action(state)
                action = (action + np.random.normal(0, exploration_noise, size=env.action_dim)*max_action)
                action = np.clip(action, min_action, max_action)
                
                next_state, reward, done, _, info = env.step(action)
                # if render and i >= render_interval : env.render()
                agent.replay_buffer.push((state, next_state, action, reward, float(done)))

                state = next_state
                if done or step >= max_length_of_trajectory:
                    break
                step += 1
                total_reward += reward
            total_step += step+1
            print("Total T:{} Episode: \t{} Total Reward: \t{:0.2f}".format(total_step, i, total_reward))
            agent.update()
           # "Total T: %d Episode Num: %d Episode T: %d Reward: %f

            # save rewards in file
            total_reward_array.append(total_reward)
            np.save('total_reward.npy', total_reward_array)

            if i % log_interval == 0:
                agent.save()
    else:
        raise NameError("mode wrong!!!")




if __name__ == '__main__':
    main()