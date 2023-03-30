import socket

def main():
    host='localhost'
    PORT = 5697

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, PORT))

    server.listen(10)
    mess ="853349"
    print("le serveur écoute sur l'adresse {host} et le port {PORT}")

    client_address = None
    while True:
        print("i")
        if client_address is None:
            client_socket, client_address = server.accept()
        if client_address is not None:
            print("connected")
            data = f"{3}{len(mess):03}{mess}"
            print(data.encode())
            client_socket.sendall(data.encode())
            #data = client_socket.recv(1024).decode('utf-8')
            #if data:
            #    print("data receive from client: {data}")

main()