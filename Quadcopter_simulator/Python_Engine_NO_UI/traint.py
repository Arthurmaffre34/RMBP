import art
import socket
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal
from torch.autograd import Variable
from torch.distributions import Categorical
import gym
import os
import errno
#import keyboard

device = torch.device("cpu")
from ascii_magic import AsciiArt
print("hey")
writer = SummaryWriter('runs/experiment_1')
os.system("clear") if os.name == 'posix' else os.system("cls")

art.tprint("RMBP", "rnd-xlarge") 
art.tprint(".inc")

#AsciiArt.from_dalle('a quadcopter', "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").to_terminal(columns=40) #dall_e

version = "0.2"

from engine import Env




class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Actor, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Sigmoid()
        )
        #self.action_scale = torch.tensor((env.action_space.high - env.action_space.low) / 2.)

    def forward(self, state):
        return self.actor(state)# * self.action_scale

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.critic(state)

def ppo(env, actor, critic, actor_optimizer, critic_optimizer, num_epochs, num_steps, gamma, lmbda, eps_clip):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    for epoch in range(num_epochs):
        state = env.reset()
        episode_rewards = []

        for t in range(num_steps):
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action = actor(state_tensor).detach().numpy()
            next_state, reward, done = env.step(action)
            print(reward)
            episode_rewards.append(reward)

            advantage, old_value = compute_advantage(critic, state_tensor, reward, next_state, done, gamma, lmbda)

            actor_loss = train_actor(actor, critic, actor_optimizer, state_tensor, action, advantage, eps_clip)
            critic_loss = train_critic(critic, critic_optimizer, state_tensor, old_value, advantage)

            writer.add_scalar('Critic loss', critic_loss, epoch)
            writer.add_scalar('actor loss', actor_loss, epoch)
            writer.add_scalar('reward', reward, epoch)

            state = next_state
            if done:
                break

        print(f"Epoch {epoch + 1}/{num_epochs}: Reward: {sum(episode_rewards)}")

def compute_advantage(critic, state, reward, next_state, done, gamma, lmbda):
    state_value = critic(state)
    next_state_value = critic(torch.tensor(next_state, dtype=torch.float32))
    target_value = torch.tensor(reward, dtype=torch.float32) + (1 - done) * gamma * next_state_value

    td_error = target_value - state_value
    advantage = td_error.detach()

    return advantage, state_value

def train_actor(actor, critic, optimizer, state, action, advantage, eps_clip):
    mu = actor(state)
    old_probs = Normal(mu, torch.tensor(1.)).log_prob(torch.tensor(action)).sum(axis=-1)
    old_probs = old_probs.detach()

    def loss_fn():
        new_mu = actor(state)
        new_probs = Normal(new_mu, torch.tensor(1.)).log_prob(torch.tensor(action)).sum(axis=-1)
        prob_ratio = torch.exp(new_probs - old_probs)
        clipped_ratio = torch.clamp(prob_ratio, 1 - eps_clip, 1 + eps_clip)
        surrogate_objective = torch.min(prob_ratio * advantage, clipped_ratio * advantage)

        return -torch.mean(surrogate_objective)
    
    optimizer.zero_grad()
    loss = loss_fn()
    loss.backward()
    optimizer.step()

    return loss
        

def train_critic(critic, optimizer, state, old_value, advantage):
    loss = torch.mean((old_value - advantage) ** 2)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


env = Env()

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
hidden_dim = 128

actor = Actor(state_dim, action_dim, hidden_dim)
critic = Critic(state_dim, hidden_dim)

actor_optimizer = optim.Adam(actor.parameters(), lr=3e-4)
critic_optimizer = optim.Adam(critic.parameters(), lr=3e-4)

num_epochs = 20000
num_steps = 2000
gamma = 0.99
lmbda = 0.95
eps_clip = 0.2



print("\nthis script is used to train a Pytorch A2C model, it makes a bridge with a TCP socket to the Unity project")
if input("initialization of the connection? (" + "\033[92m" + "y " + "\033[0m" + "or " + "\033[91m"+ "n" + "\033[0m" "): ") != "y":
    exit()

ppo(env, actor, critic, actor_optimizer, critic_optimizer, num_epochs, num_steps, gamma, lmbda, eps_clip)

