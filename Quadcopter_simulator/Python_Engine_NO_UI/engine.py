import gym
import numpy as np
import quaternion
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class Env(gym.Env): #ENV for A2C model training 
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(Env, self).__init__()

        self.observation_shape = [10, 10, 10]
        self.objective_shape = [5, 5, 5]

        #permmissible area of quad to be
        self.x_min = 0
        self.y_min = 0
        self.z_min = 0

        self.x_max = self.observation_shape[0]
        self.y_max = self.observation_shape[1]
        self.z_max = self.observation_shape[2]

        self.accel_min = -12
        self.vit_min = -12



        #init gym variable

        low_bounds = np.array([
            -12,-12,-12,-12,-12,-12,-12,-12,-12,-3,-3,-3,-3,-3,-3,-3,-3,-3
        ])

        high_bounds = np.array([
            12,12,12,12,12,12,12,12,12,3,3,3,3,3,3,3,3,3
        ])

        self.observation_space = gym.spaces.Box(low=low_bounds, high=high_bounds, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=np.array([0,0,0,0]), high=np.array([1,1,1,1]), dtype=np.float32)



        #init var to reset env
        self.rexset_pos = 1 # reset(0=no, 1=yes, 2=random pos)
        

        #command values
        self.pos_vector = np.zeros(3, dtype=float)
        self.vit_vector = np.zeros(3, dtype=float)
        self.acc_vector = np.zeros(3, dtype=float)

        self.rot_pos_vector = np.zeros(3, dtype=float)
        self.rot_vit_vector = np.zeros(3, dtype=float)
        self.rot_acc_vector = np.zeros(3, dtype=float)

        self.thrust = np.zeros(4, dtype=float)

        #physic value
        self.dt = 2e-3
        self.mass = 1 #mass is kg
        self.radius = 0.5
        self.maxthrust = 20 #maxthrust in n/s2
        self.gravity = np.array([0, 0, -10]) #m/s2

        


    def reset(self):
        if self.rexset_pos == 1: #reset postion to 0
            self.pos_vector = np.array([4, 5, 0])
            self.vit_vector = np.zeros(3, dtype=float)
            self.acc_vector = np.zeros(3, dtype=float)

            self.rot_pos_vector = np.zeros(3, dtype=float)
            self.rot_vit_vector = np.zeros(3, dtype=float)
            self.rot_acc_vector = np.zeros(3, dtype=float)

            self.thrust = np.zeros(4, dtype=float)


        elif self.rexset_pos == 2: #random position
            self.pos_vector = np.random.uniform(low=-3, high=3, size=3)
            self.vit_vector = np.random.uniform(low=-3, high=3, size=3)
            self.acc_vector = np.random.uniform(low=-3, high=3, size=3)

            self.rot_pos_vector = np.random.uniform(low=0, high=2*np.pi, size=3)
            self.rot_vit_vector = np.random.uniform(low=0, high=2*np.pi, size=3)
            self.rot_acc_vector = np.random.uniform(low=0, high=2*np.pi, size=3)

            self.thrust = np.zeros(4, dtype=float)

        else:
            pass

        initial_state = np.hstack((self.pos_vector, self.vit_vector, self.acc_vector, self.rot_pos_vector, self.rot_vit_vector, self.rot_acc_vector))
        return initial_state


    def step(self, action):
        #first set rotation
        self.rot_acc_vector = np.array([action[0] + action[3] - action[1] - action[2], action[2] + action[3] - action[0] - action[1], action[1] + action[3] - action[0] - action[2]])
        self.rot_acc_vector = (self.rot_acc_vector*self.radius)/self.mass

        self.rot_vit_vector = self.rot_vit_vector + self.rot_acc_vector * self.dt
        self.rot_pos_vector = self.rot_pos_vector + self.rot_vit_vector * self.dt
        
        #now set the position

        #transform vector using quaternion method
        thrust = action * self.maxthrust/4
        thrust_quat = np.quaternion(0, *np.array([0, 0, np.sum(thrust).item()]))

        
        rx, ry, rz = self.rot_pos_vector
        qx = quaternion.from_rotation_vector([rx, 0, 0])
        qy = quaternion.from_rotation_vector([0, ry, 0])
        qz = quaternion.from_rotation_vector([0, 0, rz])
        q = qx * qy * qz

        acc_vector_quat = q * thrust_quat * q.conjugate()

        self.acc_vector = np.array([acc_vector_quat.x, acc_vector_quat.y, acc_vector_quat.z])
        self.acc_vector = self.acc_vector + self.gravity #add gravity


        #transform the acc vector in vit and rotation
        self.vit_vector = self.vit_vector + self.acc_vector * self.dt
        self.pos_vector = self.pos_vector + self.vit_vector * self.dt


        next_state = np.hstack((self.pos_vector, self.vit_vector, self.acc_vector, self.rot_pos_vector, self.rot_vit_vector, self.rot_acc_vector))
        reward = self.reward_function(next_state)
        #done signal de fin de l'épisode à déterminer
        done = False

        return next_state, reward, done 


    def reward_function(self, state):
        position_penalty_weight = 1.0
        #position_threshold = 0.5
        
        distance = np.linalg.norm(np.array(state[:3]) - np.array(self.objective_shape)) * position_penalty_weight
        reward = -np.log(distance)
        return reward


    def close(self):
        pass


















'''

env = Env()
pos_vectors = []
vit_vectors = []

for i in range(10000):

    env.step(np.array([0.6,0.5,0.5,0.6]))
    pos_vectors.append(env.pos_vector.copy())
    vit_vectors.append(env.vit_vector.copy())

print("fin")
# Convertir en array numpy pour faciliter l'indexation
pos_vectors = np.array(pos_vectors)
vit_vectors = np.array(vit_vectors)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(pos_vectors[:, 0], pos_vectors[:, 1], pos_vectors[:, 2])

# Set the limits
ax.set_xlim([env.x_min, env.x_max])
ax.set_ylim([env.y_min, env.y_max])
ax.set_zlim([env.z_min, env.z_max])

ax.invert_yaxis()
ax.invert_xaxis()

#add objective
ax.scatter(env.objective_shape[0], env.objective_shape[1], env.objective_shape[2], color='red')

for i in range(0, len(pos_vectors), 200):
    # Normalize the velocity vector
    norm = np.linalg.norm(vit_vectors[i])
    if norm != 0: 
        normalized_vit_vector = vit_vectors[i]*2 / norm
    else:
        normalized_vit_vector = vit_vectors[i]

    # Draw an arrow from the drone's position in the direction given by the normalized velocity vector
    ax.quiver(pos_vectors[i, 0], pos_vectors[i, 1], pos_vectors[i, 2], 
              normalized_vit_vector[0], normalized_vit_vector[1], normalized_vit_vector[2],
              color='red')


ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

'''