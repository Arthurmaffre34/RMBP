import art
import socket
import time
import numpy as np
import torch
import gym
import os

from ascii_magic import AsciiArt


os.system("clear") if os.name == 'posix' else os.system("cls")

art.tprint("RMBP", "rnd-xlarge") 
art.tprint(".inc")

AsciiArt.from_dalle('a quadcopter', "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").to_terminal(columns=40) #dall_e

version = "0.1"
PORT = 5697

class Networking(): #class that use to make connection to simulator
    def __init__(self):
        #super((Networking, self).__init__())

        self.client = socket.socket(socket.self.AF_INET, socket.self.SOCK_STREAM)
        self.PORT = 5697
        self.address = "localhost"

    def connect(self, PORT, address):
        while True:
            try:
                print("connecting to server at %s", PORT)
                self.client.connect(address)
                print("server connected")
                break
            except:
                print("connection failed, retrying...")
                time.sleep(0.5)

    def send_inf_to_sim(self, client, packet): #packet is a list of information that reset the inv
        packet = str(packet).encode()
        client.send(packet)
        print("send instruction to simulator")

    def receive_inf_from_sim(self, client):
        pass
        #receive packet 




class Env(gym.Env): #ENV for A2C model training 
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super((Env, self).__init__())

        self.observation_shape = (5, 5, 5)
        self.objective_shape = (2.5, 2.5, 2.5)

        #permmissible area of quad to be
        self.x_min = 0
        self.y_min = 0
        self.z_min = 0

        self.x_max = self.observation_shape[0]
        self.y_max = self.observation_shape[1]
        self.z_max = self.observation_shape[2]

        self.x_obj = self.objective_shape[0]
        self.y_obj = self.objective_shape[1]
        self.z_obj = self.objective_shape[2]

        #init var to reset env
        self.rexset_pos = 2 # reset(0=no, 1=yes, 2=random pos)
        self.mass = 1 #mass is kg
        self.maxthrust = 2 #maxthrust in kg

    

    def reset(self, reset_pos, mass, maxthrust):
        
        Networking.send_inf_to_sim(packet=[reset_pos, mass, maxthrust]) #reset the env and the simulator

    def step(self):
        pass #make a fuction that receive the packets

    







client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
def connection(server_address = ('localhost', PORT)):
    
    while True:
        try:
            print("connecting to server at %s", PORT)
            client.connect(server_address)
            print("server connected")
            break
        except socket.error:
            print("connection failed, retrying...")
            time.sleep(0.5)




def send_inf_to_sim(packet = (1, 1, 1)): 
    #send a packet to init the simulator packet = (reset(0=no, 1=yes, 2=rdmpos), mass(kg), maxthrust(kg))
    packet = str(packet).encode()
    client.send(packet)
    print("send instruction to simulator")






print("\nthis script is used to train a Pytorch A2C model, it makes a bridge with a TCP socket to the Unity project")
if input("initialization of the connection? (" + "\033[92m" + "y " + "\033[0m" + "or " + "\033[91m"+ "n" + "\033[0m" "): ") != "y":
    exit()



#connection_test = Networking()
#connection_test.connect()
#connection()


while True:
    send_inf_to_sim()
    time.sleep(0.01)
        







