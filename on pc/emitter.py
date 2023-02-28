import socket

def emitter():
    IP = "192.168.1.30"
    PORT = 49800

    message = b"test concluant message recu"

    print("UDP target IP = %s" % IP)
    print("UDP target PORT = %s" % PORT)

    print(" message: %s" % message)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(message, (IP, PORT))
emitter()