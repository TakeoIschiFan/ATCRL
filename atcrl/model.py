import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
from matplotlib import pyplot as plt

from utils import OUActionNoise, hard_update, soft_update

torch.set_default_dtype(torch.float64)
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#problem = "atcrl:ATCEnvironment-v1"
problem = "Pendulum-v1"

# this import is required for gym to find our environment
from atcrl import ATCEnvironment
env = gym.make(problem)

num_states = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]


class ReplayBuffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size

        self.buffer_counter = 0

        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    def record(self, obs_tuple):
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1][0]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = torch.from_numpy(self.state_buffer[batch_indices])
        action_batch = torch.from_numpy(self.action_buffer[batch_indices])
        reward_batch = torch.from_numpy(self.reward_buffer[batch_indices])
        next_state_batch = torch.from_numpy(self.next_state_buffer[batch_indices])

        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        target_actions = target_actor(next_state_batch)

        y = reward_batch + gamma * target_critic([next_state_batch, target_actions.detach()])

        critic_optimizer.zero_grad()
        critic_value = critic_model([state_batch, action_batch])
        critic_loss = F.mse_loss(critic_value, y.detach())
        critic_loss.backward()
        critic_optimizer.step()

        actor_optimizer.zero_grad()
        actions = actor_model(state_batch)
        critic_value = critic_model([state_batch, actions])
        # Used `-value` as we want to maximize the value given
        # by the critic for our actions
        actor_loss = -critic_value.mean()
        actor_loss.backward()
        actor_optimizer.step()

    def __len__(self):
        return self.buffer_counter


def update_target(tau):
    soft_update(target_critic, critic_model, tau)
    soft_update(target_actor, actor_model, tau)


class Actor(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(num_states, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, num_actions),
            nn.Tanh()
        )
        self.model[-2].weight.data.uniform_(-0.003, 0.003)
        self.model[-2].bias.data.uniform_(-0.003, 0.003)

    def forward(self, inputs):
        return self.model(inputs) * upper_bound

    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))


class Critic(nn.Module):
    def __init__(self):
        super().__init__()

        self.state_model = nn.Sequential(
            nn.Linear(num_states, 16),
            nn.ReLU(),
            nn.LayerNorm(16),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.LayerNorm(32)
        )

        self.action_model = nn.Sequential(
            nn.Linear(num_actions, 32),
            nn.ReLU(),
            nn.LayerNorm(32)
        )

        self.out_model = nn.Sequential(
            nn.Linear(64, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 1)
        )

    def forward(self, inputs):
        model_input = self.state_model(inputs[0])
        action_input = self.action_model(inputs[1])

        return self.out_model(torch.cat((model_input, action_input), dim=1))

    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))


def policy(state, noise_object):
    actor_model.eval()
    sampled_actions = actor_model(state).squeeze()
    actor_model.train()

    noise = torch.from_numpy(noise_object())
    sampled_actions = sampled_actions + noise

    legal_action = sampled_actions.clamp(lower_bound, upper_bound)

    return [legal_action.squeeze().detach().numpy()]


# Hyperparameters
std_dev = 0.2
ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

actor_model = Actor()
critic_model = Critic()

target_actor = Actor()
target_critic = Critic()

target_actor.eval()
target_critic.eval()

# Making the weights equal initially
hard_update(target_actor, actor_model)
hard_update(target_critic, critic_model)

# Learning rate for actor-critic models
critic_lr = 0.002
actor_lr = 0.001

critic_optimizer = optim.Adam(critic_model.parameters(), lr=critic_lr)
actor_optimizer = optim.Adam(actor_model.parameters(), lr=actor_lr)

total_episodes = 200
# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.005

buffer = ReplayBuffer(50000, 64)

# Training loop
# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []

#print(f"we are using {torch.cuda.get_device_name(torch.cuda.current_device())}")
#actor_model.load_checkpoint("cp/actor")
#critic_model.load_checkpoint("cp/critic")

# Takes about 20 min to train
for ep in range(total_episodes):
    n = 0
    test = nn.Linear(num_states, 512)
    prev_state = env.reset()
    episodic_reward = 0
    prev_best_reward = - np.inf

    while True:
        #env.render()
        torch_prev_state = torch.tensor(prev_state, dtype=torch.float64).unsqueeze(dim=0)

        action = policy(torch_prev_state, ou_noise)
        # Recieve state and reward from environment.
        state, reward, done, info = env.step(action)

        buffer.record((prev_state, action, reward, state))
        episodic_reward += reward

        buffer.learn()
        update_target(tau)

        # End this episode when `done` is True
        n += 1
        if done or n > 500:
            break

        prev_state = state

    ep_reward_list.append(episodic_reward)

    # Mean of last 40 episodes
    avg_reward = np.mean(ep_reward_list[-40:])
    print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
    avg_reward_list.append(avg_reward)
    if avg_reward > 0:
        print("... saving models ...")
        env.render()
        actor_model.save_checkpoint("cp/actor")
        critic_model.save_checkpoint("cp/critic")
    if avg_reward > 0:
        break


# Plotting graph
# Episodes versus Avg. Rewards
plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Episodic Reward")
plt.show()
