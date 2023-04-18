import socket
import time
import random

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
        print(message)
        return message.encode('utf-8')
    
    def decode(self, data):
        mess_type = data[:1].decode('utf-8')
        longueur_contenu = int(data[1:4].decode('utf-8'))
        mess = data[4:longueur_contenu+4].decode('utf-8')
        return mess_type, mess

    def send_inf_to_sim(self, packet): #packet is a list of information that reset the inv     mess_type = 2
        packet = self.encode(2, packet)
        self.client.send(packet)
        print("send instruction to simulator")

    def send_command_to_sim(self, command):
        packet = self.encode(1, str(command)[1:-1].replace(" ", ""))
        self.client.send(packet)
        print("send command to simulator")

    def receive_inf_from_sim(self):
        data = None
        try:
            data = self.client.recv(102400, socket.MSG_DONTWAIT)
            print(data)
        except:
            pass
        if data:
            # Vider le tampon de réception
            while True:
                try:
                    self.client.recv(1024, socket.MSG_DONTWAIT)
                    print("bam")
                except:
                    break
            mess_type, mess = self.decode(data)
            if mess_type == "3":
                sub_mess = mess.split(",")
                self.fly_inf = []
                for submess in sub_mess:
                    self.fly_inf.append(float(submess))

            return self.fly_inf
        return None, None
    
connection = Networking()
connection.connect()


#command = [0.5,0.5,0.7,0.5]
while True:
    connection.receive_inf_from_sim()
    print(connection.fly_inf)
    connection.send_inf_to_sim('1')
    connection.send_command_to_sim('1')
    time.sleep(1)