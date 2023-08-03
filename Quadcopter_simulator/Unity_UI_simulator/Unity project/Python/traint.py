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

class Networking: #class that use to make connection to simulator

    #mess type 1 = command train to simulator, 2 = reset state function train to simulator, 3 = informations from simulator to train

    def __init__(self, address = "localhost", PORT = 5697): #default adress localhost, default port 5697
        #super((Networking, self).__init__())

        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.PORT = PORT
        self.address = address
        self.fly_inf = []

    def connect(self, PORT=None, address=None):
        address = address or self.address
        PORT = PORT or self.PORT
        while True:
            try:
                print("connecting to server at %s", PORT)
                print(address, PORT)
                self.client.connect((address, PORT))
                print("server connected")
                break
            except:
                print("connection failed, retrying...")
                
                time.sleep(0.5)

    def close(self):
        self.client.close()

    def encode(self, mess_type, mess): #mess type 1 = command, 2 = reset fonction
        longueur_contenu = len(mess)
        message = f"{mess_type}{longueur_contenu:03}{mess}"
        #print(message)
        return message.encode('utf-8')
    
    def decode(self, data):
        mess_type = data[:1].decode('utf-8')
        longueur_contenu = int(data[1:4].decode('utf-8'))
        mess = data[4:longueur_contenu+4].decode('utf-8')
        return mess_type, mess

    def send_inf_to_sim(self, packet): #packet is a list of information that reset the inv     mess_type = 2
        packet = self.encode(2, packet)
        self.client.send(packet)
        #print("send instruction to simulator")

    def send_command_to_sim(self, command):
        list = ""
        for i in command:
            list = list + str(i) + ","
        packet = self.encode(1, list[:-1])
        self.client.send(packet)


    def receive_inf_from_sim(self): #take only the last packet from env
        try:
            self.client.recv(102400, socket.MSG_DONTWAIT)
        except:
            pass
        data = self.client.recv(102400)
        mess_type, mess = self.decode(data) #decoder 
        if mess_type == "3":
            sub_mess = mess.split(",")
            self.fly_inf = []
            for submess in sub_mess:
                self.fly_inf.append(float(submess))

        return self.fly_inf




class Env(gym.Env): #ENV for A2C model training 
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(Env, self).__init__()
        self.connection = Networking()
        self.connection.connect()

        self.observation_shape = [10, 10, 10]
        self.objective_shape = [5, 5, 5]

        low_bounds = np.array([
            -12,-12,-12,-12,-12,-12,-12,-12,-12,-3,-3,-3,-3,-3,-3,-3,-3,-3
        ])

        high_bounds = np.array([
            12,12,12,12,12,12,12,12,12,3,3,3,3,3,3,3,3,3
        ])

        self.observation_space = gym.spaces.Box(low=low_bounds, high=high_bounds, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=np.array([0,0,0,0]), high=np.array([1,1,1,1]), dtype=np.float32)

        #init var to reset env
        self.rexset_pos = 2 # reset(0=no, 1=yes, 2=random pos)
        self.mass = 1 #mass is kg
        self.maxthrust = 2 #maxthrust in kg

    def reset(self):
        time.sleep(0.01)
        self.connection.send_inf_to_sim([1, self.mass, self.maxthrust]) #reset the env and the simulator
        time.sleep(0.01)
        initial_state = np.array(self.connection.receive_inf_from_sim())
        return initial_state

    def step(self, action):
        action = np.clip(action, -1, 1)
        self.connection.send_command_to_sim((action+1)/2)
        self.next_state = (np.array(self.connection.receive_inf_from_sim())+12)/12
        reward = self.reward_function(self.next_state)
        #done signal de fin de l'épisode à déterminer
        done = False
        info = False

        return self.next_state, reward, done, info


    def reward_function(self, state):
        position_penalty_weight = 1.0
        #position_threshold = 0.5
        
        distance = np.linalg.norm(np.array(state[:3]) - np.array(self.objective_shape)) * position_penalty_weight
        reward = -np.log(distance)
        return reward

    def render(self, mode='human'):
        print(f"Current state: {self.next_state}")

    def close(self):
        self.connection.close()






class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Actor, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        self.action_scale = torch.tensor((env.action_space.high - env.action_space.low) / 2.)

    def forward(self, state):
        return self.actor(state) * self.action_scale

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
            next_state, reward, done, _ = env.step(action)
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
num_steps = 400
gamma = 0.99
lmbda = 0.95
eps_clip = 0.2



print("\nthis script is used to train a Pytorch A2C model, it makes a bridge with a TCP socket to the Unity project")
if input("initialization of the connection? (" + "\033[92m" + "y " + "\033[0m" + "or " + "\033[91m"+ "n" + "\033[0m" "): ") != "y":
    exit()

ppo(env, actor, critic, actor_optimizer, critic_optimizer, num_epochs, num_steps, gamma, lmbda, eps_clip)

