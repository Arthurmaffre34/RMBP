import numpy as np
import matplotlib.pyplot as plt

class PIDController:
    def __init__(self, kp=0.16, ki=0.07, kd=0.06, dt=2e-3):
        self.kp = kp #gain proportionnel
        self. ki = ki #gain intégral
        self.kd = kd #gain dérivé
        self.dt = dt #delta temps
        self.integral_error = 0.0 #accumulation pour l'erreur intégrale
        self.previous_error = 0.0 #stockage de l'erreur précédente

    def control(self, setpoint, measurement):
        #calculer erreur actuelle
        error = setpoint - measurement

        #calculer valeur proportionelle
        proportional = self.kp * error

        #calculer la valeur intégrale
        self.integral_error += error * self.dt
        integral = self.ki * self.integral_error

        #calculer la composante dérivée
        derivate_error = (error - self.previous_error) / self.dt
        derivate = self.kd * derivate_error

        #mise a jour de l'erreur précédente
        self.previous_error = error

        #calculer la sortie de controle
        u = proportional + integral + derivate
        return u 


class DroneController:
    def __init__(self):
        self.pid_pos = [PIDController() for _ in range(3)]
        self.pid_vit = [PIDController() for _ in range(3)]
        self.pid_acc = [PIDController() for _ in range(3)]

        self.pid_yaw = PIDController(0.1,0.048,0.1)
        self.pid_pitch = PIDController(4.5,0.048,1.5)
        self.pid_roll = PIDController(4.5,0.048,1.5)

    def control(self, state, position_target, orientation_target, dt):
        
        #calcul erreur de position
        print(state[:3])
        error_pos = position_target - state[:3]
        print("error y", error_pos[1])
        
        #calcul du set point de la vitesse par le pid
        speed_setpoint = list(map(lambda pid_pos_p, pos_t, pos_a: pid_pos_p.control(pos_t, pos_a), *zip(self.pid_pos, position_target, state[:3])))
        print(speed_setpoint)
        #calcul erreur d'orientation
        error_yaw = orientation_target[2] - state[9]
        error_pitch = orientation_target[1] - state[10]
        error_roll = orientation_target[0] - state[11]
        print("error roll", error_roll)
        
        #commanse pour les moteurs basées sur les erreurs de position
        motor_command_x = self.pid_x.control(error_x, dt)
        motor_command_y = self.pid_y.control(error_y, dt)
        motor_command_z = self.pid_z.control(error_z, dt)

        #commande pour les moteurs basée sur l'erreur d'orientation
        motor_command_yaw = self.pid_yaw.control(error_yaw, dt)
        motor_command_pitch = self.pid_pitch.control(error_pitch, dt)
        motor_command_roll = self.pid_roll.control(error_roll, dt)
        #motor_command_pitch, motor_command_roll = 0,0


        #aggrégation des commandes pour chaque moteur
        print (motor_command_x, motor_command_pitch)
        motor_command_0 = motor_command_z - motor_command_x - motor_command_y + motor_command_yaw - motor_command_pitch + motor_command_roll
        motor_command_1 = motor_command_z - motor_command_x + motor_command_y - motor_command_yaw - motor_command_pitch - motor_command_roll
        motor_command_2 = motor_command_z + motor_command_x + motor_command_y + motor_command_yaw + motor_command_pitch - motor_command_roll
        motor_command_3 = motor_command_z + motor_command_x - motor_command_y - motor_command_yaw + motor_command_pitch + motor_command_roll

        
        motor_command = np.array([motor_command_0, motor_command_1, motor_command_2, motor_command_3], dtype=float)
        motor_command = np.clip(motor_command, np.zeros(4), np.array([1,1,1,1]))
        print(motor_command)
        return motor_command


drone_controller = DroneController()

from engine import Env

env = Env()





pos_vectors = []
vit_vectors = []

state = env.reset()
for i in range(20000):
    print(i)
    motor_command = drone_controller.control(state, env.objective_shape, np.zeros(3), 2e-3)
    env.step(motor_command)
    #env.step(np.array([1,1,0.5,0.5]))
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

for i in range(0, len(pos_vectors), 100):
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