import socket

# Tello's address and port for commands
tello_address = ('192.168.10.1', 8889)

# Create a UDP socket and bind to a local port (e.g., 9000)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('', 9000))

# Function to send a command and print the response
def send_command(command):
    try:
        sock.sendto(command.encode('utf-8'), tello_address)
        response, _ = sock.recvfrom(1024)
        print(f"Response: {response.decode('utf-8')}")
    except Exception as e:
        print(f"Error: {e}")

# Enter SDK mode by sending the 'command' string
while True:
    command_input = input("Command: ")
    if command_input == 'exit':
        break
    send_command(command_input)
