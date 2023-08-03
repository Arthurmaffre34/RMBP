import socket
import time
import random
import errno

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
        print(data)
        mess_type, mess = self.decode(data) #decoder 
        if mess_type == "3":
            sub_mess = mess.split(",")
            self.fly_inf = []
            for submess in sub_mess:
                self.fly_inf.append(float(submess))

        return self.fly_inf



    
connection = Networking()
connection.connect()


#command = [0.5,0.5,0.7,0.5]
while True:
    connection.send_inf_to_sim('a')
    print('reset')
    time.sleep(0.01)
    for i in range (100):
        connection.send_command_to_sim([i/100,i/100,i/100,i/100])
        time.sleep(0.01)




class Actor(nn.Module): #a revoir pour le max_action
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, state):
        return self.network(state)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        return self.network(torch.cat([state, action], dim=-1))


class DDPGAgent:
    def __init__(self, state_dim, action_dim, hidden_dim, actor_lr, critic_lr, gamma, tau):
        self.actor = Actor(state_dim, hidden_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, hidden_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict()) #load same model
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.tau = tau

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

    def choose_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.actor(state).data.cpu().numpy().flatten()
        return action

    def train(self, replay_buffer, batch_size):
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)#.unsqueeze(-1)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(1-dones).to(device)

        #print(next_states[1])

        next_actions = self.actor_target(next_states)
        
        next_q_values = self.critic_target(next_states, next_actions)
        #print(next_q_values[1])
        target_q_values = rewards +  self.gamma * next_q_values *(dones).detach()
        #print(target_q_values[1])


        q_values = self.critic(states, actions)
        #print(q_values[1])
        #print(rewards[1])
        #optimize the critic
        critic_loss = nn.MSELoss()(q_values, target_q_values)
        critic_losss = critic_loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        #print(self.critic(states, actions).mean())
        #compute actor loss
        actor_loss = -self.critic(states, self.actor(states)).mean()
        actor_losss = actor_loss
        self.actor_optimizer.zero_grad
        actor_loss.backward()
        self.actor_optimizer.step()
        return critic_losss, actor_losss

    '''
    def save(self):
        #torch.save(self.actor.state_dict(), directory + 'actor.pth')
        #torch.save(self.critic.state_dict(), directory + 'critic.pth')
    def load(self):
        self.actor.load_state_dict(torch.load(directory + 'actor.pth'))
        self.critic.load_state_dict(torch.load(directory + 'critic.pth'))
    '''




class ReplayBuffer: #### finito pipo
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []
        self.index = 0

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) == self.max_size:
            self.buffer[int(self.index)] = (state, action, reward, next_state, done)
            self.index = (self.index + 1) % self.max_size
            print("full")
        else:
            self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        idxs = np.random.randint(0, len(self.buffer), size=batch_size)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in idxs:
            state, action, reward, next_state, done = self.buffer[i]
            states.append(np.array(state, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(np.array(reward, copy=False))
            next_states.append(np.array(next_state, copy=False))
            dones.append(np.array(done, copy=False))

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards).reshape(-1, 1)
        next_states = np.array(next_states)
        dones = np.array(dones).reshape(-1, 1)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)





def trainDDPG():
    # Hyperparameters
    n_episodes = 5000
    batch_size = 64
    start_training_after = 0
    replay_buffer_size = 5000
    exploration_noise = 0.01
    time_epoch = 2 #time of epoch in s

    # Create environment
    env = Env()

    # Create DDPG agent

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    hidden_dim = 256
    actor_lr = 0.01
    critic_lr = 0.01
    gamma = 0.9
    tau = 0.05

    agent = DDPGAgent(state_dim, action_dim, hidden_dim, actor_lr, critic_lr, gamma, tau)

    # Create replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size)

    # Training loop
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        iteration_count = 0
        start_time = time.time() #enregistre le temps de début de l'épisode
        while not done:
            # Choose action with exploration noise
            action = agent.choose_action(state)
            
            action = action + np.random.normal(0, exploration_noise, size=4)
            action = np.clip(action, 0, 1)
            # Perform action and store experience in replay buffer
            next_state, reward, done, _ = env.step(action)
            replay_buffer.add(state, action, reward, next_state, np.float(done))

            
            # Train agent if enough experience is stored in the replay buffer
            if len(replay_buffer) > start_training_after:
                critic_loss, actor_loss = agent.train(replay_buffer, batch_size)
                

            # Soft update target networks
            for target_param, param in zip(agent.actor_target.parameters(), agent.actor.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for target_param, param in zip(agent.critic_target.parameters(), agent.critic.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            
            state = next_state
            episode_reward += reward

            time.sleep(0.001)
            elapsed_time = time.time() - start_time #calculer le temps écoulé depuis le début de l'épisode
            if elapsed_time >= time_epoch: #si le temps est écoulé
                break

            # Incrémenter le compteur d'itérations
            iteration_count += 1

            # Calculer le temps écoulé et les itérations par seconde
            elapsed_time = time.time() - start_time
            iterations_per_second = iteration_count / elapsed_time

            # Afficher les itérations par seconde
            #print(f"Iterations per second: {iterations_per_second:.2f}")

        print(f"Episode {episode}: Reward = {episode_reward}")
        # Log reward in TensorBoard
        writer.add_scalar("Episode Reward", episode_reward, episode)
        writer.add_scalar('Actor loss', actor_loss, episode)
        writer.add_scalar('Critic loss', critic_loss, episode)



'''
def train():
    #paramètres
    num_episodes = 1000
    time_epoch = 2 #time of epoch in s
    input_size = 18 #remplacer par taille de l'état
    hidden_size = 2560
    output_size = 4 #remplacer par le nombre d'actions possibles
    learning_rate = 1e-2

    #Initialisation de l'environnement et du modèle PPO

    env=Env()
    model = PPO(input_size, hidden_size, output_size, learning_rate=learning_rate)

    print("boucle d'entrainement appuyer sur s pour sauvegarder, p pour pause et e pour echap")
    #Boucle d'entrainement
    for episode in range(num_episodes):
        state = env.reset()
        states, actions, rewards, next_states, dones = [], [], [], [], []
        done = False

        start_time = time.time() #enregistre le temps de début de l'épisode
        while not done:
            #Sélection de l'action à partir de l'état actuel en utilisant l'acteu
            action = model.actor(torch.FloatTensor(np.array(state)).to(device)).detach().numpy()

            #Exécution de l'action et obtention de l'état suivant, de la récompense et du signal 'done'
            next_state, reward, done = env.step((action+1)/2)

            #stockage des données pour l'entrainement
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

            #Mise à jour de l'état
            state = next_state

            elapsed_time = time.time() - start_time #calculer le temps écoulé depuis le début de l'épisode
            if elapsed_time >= time_epoch: #si le temps est écoulé
                break

        
        #Entrainement du modèle A2C avec les données collectées
        actor_loss, critic_loss = model.train(states, actions, rewards, next_states, dones)

        #écrire les pertes dans tensorboard
        writer.add_scalar('Actor loss', actor_loss, episode)
        writer.add_scalar('Critic loss', critic_loss, episode)
        writer.add_scalar('Reward', sum(rewards)/len(rewards), episode)


        #affichage des informations
        print(f"Episode: {episode + 1}, Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}")
    writer.close()


    

'''