from socket import socket, AddressFamily, SocketKind
import random

server_socket = socket(AddressFamily.AF_INET, SocketKind.SOCK_DGRAM)
server_socket.bind(('0.0.0.0', 8889))

while True:
    data, addr = server_socket.recvfrom(1024)  # This is a blocking call
    print("Received message:", data, "from", addr)

server_socket.close()
