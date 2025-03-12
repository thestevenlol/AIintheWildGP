import socket

UDP_IP = "192.168.10.1"
UDP_PORT = 8889

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

while True:
    message = input("Enter message to send: ")
    sock.sendto(message.encode(), (UDP_IP, UDP_PORT))
    data, addr = sock.recvfrom(1024)
    print("received message: %s" % data)