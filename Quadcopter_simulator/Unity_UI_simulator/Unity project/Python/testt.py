import art
import socket
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal
import gym
import os
#import keyboard

device = torch.device("mps")
from ascii_magic import AsciiArt
print("hey")
writer = SummaryWriter('runs/experiment_1')

os.system("clear") if os.name == 'posix' else os.system("cls")

art.tprint("RMBP", "rnd-xlarge") 
art.tprint(".inc")

#AsciiArt.from_dalle('a quadcopter', "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").to_terminal(columns=40) #dall_e

version = "0.1"

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

    def receive_inf_from_sim(self):
        data = None
        try:
            data = self.client.recv(102400, socket.MSG_DONTWAIT)
            #print(data)
        except:
            pass
        if data:
            # Vider le tampon de réception
            while True:
                try:
                    self.client.recv(1024, socket.MSG_DONTWAIT)
                    #print("bam")
                except:
                    break
            mess_type, mess = self.decode(data)
            if mess_type == "3":
                sub_mess = mess.split(",")
                self.fly_inf = []
                for submess in sub_mess:
                    self.fly_inf.append(float(submess))

            return self.fly_inf
        return self.fly_inf




class Env(gym.Env): #ENV for A2C model training 
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(Env, self).__init__()
        self.connection = Networking()
        self.connection.connect()

        self.observation_shape = [10, 10, 10]
        self.objective_shape = [5, 5, 5]

        #permmissible area of quad to be
        self.x_min = 0
        self.y_min = 0
        self.z_min = 0

        self.accel_min = -12
        self.vit_min = -12

        #init var to reset env
        self.rexset_pos = 2 # reset(0=no, 1=yes, 2=random pos)
        self.mass = 1 #mass is kg
        self.maxthrust = 2 #maxthrust in kg

        

    

    def reset(self):
        self.connection.send_inf_to_sim([1, self.mass, self.maxthrust]) #reset the env and the simulator
        time.sleep(0.01)
        initial_state = np.array(self.connection.receive_inf_from_sim())
        return initial_state

    def step(self, action):
        self.connection.send_command_to_sim(0.5,0.5,0.5,0.5)
        #self.connection.send_command_to_sim(action)
        next_state = (np.array(self.connection.receive_inf_from_sim())+12)/12
        print(next_state)
        reward = self.reward_function(next_state)

        #done signal de fin de l'épisode à déterminer
        done = False

        return next_state, reward, done 


    def reward_function(self, state):
        position_penalty_weight = 1.0
        #position_threshold = 0.5
        
        distance = np.linalg.norm(np.array(state[:3]) - np.array(self.objective_shape)) * position_penalty_weight
        reward = 1/ (distance + 1)
        return (reward+1)**2


    def close(self):
        self.connection.close()











# Define the policy network
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.mu = nn.Linear(64, action_dim)
        self.log_std = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mu = torch.sigmoid(self.mu(x))
        log_std = self.log_std(x)
        return mu, log_std

# Define the value network
class ValueNetwork(nn.Module):
    def __init__(self, input_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        value = self.fc3(x)
        return value



# Create the environment
env = Env() # You can replace this with your own environment

# Initialize networks, optimizer, and other hyperparameters
policy = PolicyNetwork(18, 4)
value = ValueNetwork(18)
policy_optimizer = optim.Adam(policy.parameters(), lr=3e-4)
value_optimizer = optim.Adam(value.parameters(), lr=1e-3)
num_epochs = 2000
num_steps = 2048
num_mini_batches = 64
clip_epsilon = 0.3
gamma = 0.95
gae_lambda = 0.95

# Collect trajectories
def collect_trajectories(env, policy, num_steps):
    states = []
    actions = []
    rewards = []
    dones = []
    state = env.reset()
    for _ in range(num_steps):
        mu, log_std = policy(torch.tensor(state, dtype=torch.float32))
        action = torch.normal(mu, log_std.exp()).detach().numpy()
        next_state, reward, done = env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        state = next_state
        if done:
            state = env.reset()
    return np.array(states), np.array(actions), np.array(rewards), np.array(dones)

# Update the policy and value networks
def update(states, actions, rewards, dones, policy, value, policy_optimizer, value_optimizer, num_mini_batches, clip_epsilon, gamma, gae_lambda):
    # Compute advantages and returns
    advantages = np.zeros_like(rewards)
    returns = np.zeros_like(rewards)
    gae = 0
    next_value = 0 if dones[-1] else value(torch.tensor(states[-1], dtype=torch.float32)).item()
    for t in reversed(range(num_steps)):
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - value(torch.tensor(states[t], dtype=torch.float32)).item()
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages[t] = gae
        returns[t] = gae + value(torch.tensor(states[t], dtype=torch.float32)).item()
        next_value = value(torch.tensor(states[t], dtype=torch.float32)).item()

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Convert data to tensors
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.float32)
    returns = torch.tensor(returns, dtype=torch.float32).unsqueeze(1)
    advantages = torch.tensor(advantages, dtype=torch.float32).unsqueeze(1)

    # Perform PPO update
    for _ in range(num_mini_batches):
        # Update value network
        value_optimizer.zero_grad()
        value_loss = ((value(states) - returns) ** 2).mean()
        value_loss.backward()
        value_optimizer.step()

        # Log value loss to TensorBoard
        writer.add_scalar("Loss/Value", value_loss.item(), epoch)

        # Update policy network
        policy_optimizer.zero_grad()
        mu, log_std = policy(states)
        log_prob = torch.distributions.Normal(mu, log_std.exp()).log_prob(actions).sum(dim=1, keepdim=True)
        old_log_prob = log_prob.detach()
        ratio = torch.exp(log_prob - old_log_prob)
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
        policy_loss = -torch.min(surrogate1, surrogate2).mean()
        policy_loss.backward()
        policy_optimizer.step()

        # Log policy loss to TensorBoard
        writer.add_scalar("Loss/Policy", policy_loss.item(), epoch)

# Train the PPO agent
for epoch in range(num_epochs):
    states, actions, rewards, dones = collect_trajectories(env, policy, num_steps)
    update(states, actions, rewards, dones, policy, value, policy_optimizer, value_optimizer, num_mini_batches, clip_epsilon, gamma, gae_lambda)
    # Log reward to TensorBoard
    writer.add_scalar("Reward", rewards.sum(), epoch)

    print(f"Epoch {epoch + 1}/{num_epochs}: Reward = {rewards.sum()}")