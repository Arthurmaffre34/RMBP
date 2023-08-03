import socket
import time

def main():
    host='localhost'
    PORT = 5697

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    server.bind((host, PORT))

    server.listen(10)
    mess ="123"
    print("le serveur écoute sur l'adresse {host} et le port {PORT}")

    client_socket, client_address = server.accept()
    print("connected")
    
    while True:
        
        data = f"{3}{len(mess):03}{mess}"
        print(data.encode())
        client_socket.sendall(data.encode())
        time.sleep(0.5)
        """try:
            data = None
            data = client_socket.recv(1024).decode('utf-8')
            print(data)
        except:
            continue
"""
main()