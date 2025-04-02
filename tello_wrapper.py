import socket
import cv2
import numpy as np

# Tello IP and port (default)
tello_ip = '192.168.10.1'
tello_port = 8889
tello_address = (tello_ip, tello_port)

# Local UDP server address to receive video stream
local_host = '0.0.0.0' # Listen on all interfaces
local_port = 11111
local_address = (local_host, local_port)

# Create UDP sockets
command_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
video_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Bind video socket to local address
video_socket.bind(local_address)

def send_command(command):
    """Sends a command to the Tello drone and prints the response."""
    try:
        command_socket.sendto(command.encode('utf-8'), tello_address)
        response, ip_address = command_socket.recvfrom(1024)
        print(f"Tello: {response.decode()}")
        return response.decode()
    except socket.error as e:
        print(f"Socket error: {e}")
        return None

# Initialize Tello and start video stream
send_command("command")  # Enter command mode
send_command("streamon") # Enable video stream

frame = None  # Initialize frame variable

try:
    while True:
        try:
            data, addr = video_socket.recvfrom(2048) # Buffer size can be adjusted
            if len(data) > 0:
                # Decode H.264 frame using OpenCV
                nparr = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is not None and frame.shape != (): # Check if frame is valid
                    cv2.imshow("Tello Video Stream", frame)

            key = cv2.waitKey(1) & 0xFF # Small delay and get key press
            if key == ord('q'): # Press 'q' to quit
                break

        except socket.timeout as e:
            print(f"Socket timeout: {e}")
        except Exception as e:
            print(f"Error receiving data: {e}")
            break

finally:
    print("Exiting...")
    send_command("streamoff") # Turn off video stream
    command_socket.close()
    video_socket.close()
    cv2.destroyAllWindows()