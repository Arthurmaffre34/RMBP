import socket #send UDP request, send for telemetry and receive for control

def receive():
    PORT = 49800

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('', PORT))

    print("en attente de message:")

    while True:
        data, addr = sock.recvfrom(1024)
        print("received message %s" % data)

receive()

