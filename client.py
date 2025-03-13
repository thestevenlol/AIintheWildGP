import socket
import time
import sys
import threading
import cv2
import subprocess
import os
from djitellopy import Tello
from tello_stream import TelloVideoStream

def check_network_connection(host='192.168.10.1'):
    """Check if we can ping the Tello's IP address"""
    try:
        print(f"Checking connection to Tello at {host}...")
        # Use subprocess to ping with timeout
        result = subprocess.run(
            ['ping', '-c', '1', '-W', '1', host],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        if result.returncode == 0:
            print(f"Successfully pinged {host}")
            return True
        else:
            print(f"Could not ping {host}. Is the drone turned on?")
            print("Make sure your computer is connected to the Tello's WiFi network.")
            return False
    except Exception as e:
        print(f"Error checking network: {e}")
        return False

def get_distance(command: str):
    try:
        distance = command.split()[1]
        distance = int(distance)
        if distance < 20:
            print("Distance must be at least 20 cm.")
            return None
        
        if distance > 500:
            print("Distance must be at most 500 cm.")
            return None
        
        return distance
    except (ValueError, IndexError):
        print("Invalid distance format.")
        return None

def video_display_thread(tello_stream):
    """Run the video display in a separate thread"""
    if tello_stream.open_video_stream():
        try:
            print("Video stream opened successfully")
            while tello_stream.is_running:
                frame = tello_stream.read_frame()
                if frame is not None:
                    # Display the frame
                    tello_stream.display_frame(frame)
                    
                # Check for exit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    tello_stream.is_running = False
                    break
                    
                time.sleep(0.03)  # Small delay to prevent high CPU usage
        except Exception as e:
            print(f"Error in video thread: {e}")
    else:
        print("Failed to open video stream")

def main():
    # First check if we can reach the Tello
    if not check_network_connection():
        print("Network connectivity issues detected.")
        if input("Would you like to try to connect anyway? (y/n): ").lower().strip() != 'y':
            print("Exiting program.")
            return
    
    # Initialize the Tello object with increased retry count and timeout
    try:
        print("Initializing Tello drone...")
        tello = Tello()
        tello.RETRY_COUNT = 5  # Increase retry count
        tello.RESPONSE_TIMEOUT = 15  # Increase timeout in seconds
        tello.connect()
        print(f"Connection successful! Battery level: {tello.get_battery()}%")
    except Exception as e:
        print(f"Failed to initialize Tello: {e}")
        if input("Would you like to try to continue anyway? (y/n): ").lower().strip() != 'y':
            print("Exiting program.")
            return
        return
        
    # Initialize the video stream
    tello_stream = TelloVideoStream()
    
    # Start video stream
    print("Starting video stream...")
    tello.streamon()  # Use the Tello object to start streaming
    
    if tello_stream.connect():
        # Start video display in a separate thread
        tello_stream.is_running = True
        video_thread = threading.Thread(target=video_display_thread, args=(tello_stream,))
        video_thread.daemon = True
        video_thread.start()
        
        print("Video stream started. Press 'q' in the video window to stop.")
        print("\nCommand interface ready. Type 'exit' to quit.")
        print("You can enter Tello SDK commands directly (e.g., 'takeoff', 'land', 'battery?')")
        
        # Command loop
        while True:
            try:
                command_in = input("\nEnter command: ")
                if command_in.lower() == "exit":
                    print("Exiting program...")
                    try:
                        tello.land()  # Make sure the drone lands if flying
                    except:
                        pass
                    break
                
                # Special handling for certain commands
                if command_in.startswith("forward"):
                    distance = get_distance(command_in)
                    if distance is None:
                        continue
                
                # Send command to Tello using the library
                try:
                    response = tello.send_command_with_return(command_in)
                    print(f"Response: {response}")
                except Exception as e:
                    print(f"Command failed: {e}")
            except KeyboardInterrupt:
                print("\nProgram interrupted")
                break
        
        # Cleanup
        print("Stopping video stream...")
        tello_stream.is_running = False
        tello_stream.stop()
        tello.streamoff()  # Stop the stream using the Tello object
        
        print("Program terminated.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {e}")
        cv2.destroyAllWindows()  # Ensure windows are closed