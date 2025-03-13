import cv2
import os
import time
import socket
import numpy as np

# Force OpenCV to use XWayland for better compatibility on Linux
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["GDK_BACKEND"] = "x11"

class TelloVideoStream:
    """
    A class to handle video streaming from a Tello drone.
    Uses direct socket connection to the drone to receive and display video frames.
    """
    def __init__(self, tello_ip='192.168.10.1', tello_port=8889, command_port=9000):
        """
        Initialize the TelloVideoStream object.
        
        Args:
            tello_ip (str): IP address of the Tello drone
            tello_port (int): Command port of the Tello drone
            command_port (int): Local port to bind the command socket to
        """
        self.tello_ip = tello_ip
        self.tello_port = tello_port
        self.command_port = command_port
        self.window_name = 'Tello Video Stream'
        self.is_running = False
        self.cap = None
        self.use_socket = False  # Flag to indicate whether to use our own socket or not
        
        # We'll only create a command socket if explicitly needed
        self.command_socket = None

    def create_command_socket(self):
        """Create a command socket when needed"""
        if self.command_socket is None:
            self.command_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.command_socket.bind(('', self.command_port))
            print(f"Created command socket on port {self.command_port}")

    def send_command(self, command):
        """
        Send a command to the Tello drone and wait for a response.
        Only use this if needed - prefer using djitellopy for commands.
        
        Args:
            command (str): Command to send to the drone
            
        Returns:
            str: Response from the drone or None if timed out
        """
        self.create_command_socket()
        
        self.command_socket.sendto(command.encode('utf-8'), (self.tello_ip, self.tello_port))
        print(f"[TelloVideoStream] Sent: {command}")
        
        # Wait for response
        try:
            self.command_socket.settimeout(2)
            response, _ = self.command_socket.recvfrom(1024)
            response_text = response.decode('utf-8')
            print(f"[TelloVideoStream] Response: {response_text}")
            return response_text
        except socket.timeout:
            print("[TelloVideoStream] No response received (timeout)")
            return None
        finally:
            time.sleep(0.1)  # Reduced delay for better responsiveness

    def connect(self, use_socket=False):
        """
        Initialize the video display without sending stream commands.
        Assumes the Tello's video stream is already started by djitellopy.
        
        Args:
            use_socket (bool): Whether to use the socket for commands or not
            
        Returns:
            bool: True if successful, False otherwise
        """
        self.use_socket = use_socket
        
        # If using socket for commands, we need to start the stream
        if self.use_socket:
            print("[TelloVideoStream] Starting video stream...")
            stream_response = self.send_command("streamon")
            if stream_response != "ok":
                print("[TelloVideoStream] Failed to start video stream")
                return False
                
            time.sleep(1)  # Give time for video stream to start
        
        # Create a cv2 window
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            print("[TelloVideoStream] Created display window")
            return True
        except Exception as e:
            print(f"[TelloVideoStream] Failed to create window: {e}")
            return False

    def open_video_stream(self):
        """
        Open the video stream from the Tello drone.
        
        Returns:
            bool: True if successful, False otherwise
        """
        # Initialize VideoCapture with specific options for reliable UDP streaming
        self.cap = cv2.VideoCapture()
        
        # Set advanced options
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 10)  # Increase buffer size
        
        # Open with custom parameters - ensure resilience
        # Format: 'udp://@0.0.0.0:11111'
        stream_url = f'udp://@0.0.0.0:11111?overrun_nonfatal=1&fifo_size=5000000'
        tries = 0
        max_tries = 3
        
        while tries < max_tries:
            success = self.cap.open(stream_url, cv2.CAP_FFMPEG)
            if success and self.cap.isOpened():
                print(f"[TelloVideoStream] Successfully opened video stream at {stream_url}")
                return True
            
            print(f"[TelloVideoStream] Failed to open video stream, attempt {tries+1}/{max_tries}")
            time.sleep(1)
            tries += 1
            
        print(f"[TelloVideoStream] Failed to open video stream after {max_tries} attempts")
        return False

    def read_frame(self):
        """
        Read a frame from the video stream.
        
        Returns:
            numpy.ndarray or None: Frame if available, None otherwise
        """
        if self.cap is None or not self.cap.isOpened():
            return None
            
        ret, frame = self.cap.read()
        if ret:
            # Resize for stability
            try:
                return cv2.resize(frame, (640, 480))
            except Exception as e:
                print(f"[TelloVideoStream] Error processing frame: {e}")
                return None
        return None

    def display_frame(self, frame):
        """
        Display a frame in the CV2 window.
        
        Args:
            frame (numpy.ndarray): Frame to display
        """
        if frame is not None:
            cv2.imshow(self.window_name, frame)

    def start_video_display(self):
        """
        Start displaying video frames from the Tello drone.
        This is a blocking function that runs until 'q' is pressed.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.open_video_stream():
            return False
            
        self.is_running = True
        
        try:
            while self.is_running:
                frame = self.read_frame()
                if frame is not None:
                    self.display_frame(frame)
                    
                    # Exit on 'q' key
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\n[TelloVideoStream] Exiting video display - 'q' key pressed")
                        break
                else:
                    print("[TelloVideoStream] Waiting for video frame...", end="\r")
                    time.sleep(0.1)  # Small delay when no frame is available
                    
        except KeyboardInterrupt:
            print("\n[TelloVideoStream] Program interrupted by user")
        finally:
            return True
    
    def stop(self):
        """
        Stop the video stream and release resources.
        """
        self.is_running = False
        
        # Stop video stream if we started it
        if self.use_socket and self.command_socket:
            try:
                self.send_command("streamoff")
            except:
                pass  # Ignore errors when shutting down
        
        # Release resources
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        # Close socket
        if self.command_socket:
            try:
                self.command_socket.close()
            except:
                pass
            self.command_socket = None
        
        # Close OpenCV windows
        cv2.destroyAllWindows()
        print("[TelloVideoStream] Resources released, video stream stopped")

# Example usage
if __name__ == "__main__":
    tello_stream = TelloVideoStream()
    try:
        # When run directly, use socket commands
        # First send command to enter SDK mode
        tello_stream.create_command_socket()
        response = tello_stream.send_command("command")
        if response == "ok":
            if tello_stream.connect(use_socket=True):
                tello_stream.start_video_display()
    except Exception as e:
        print(f"[TelloVideoStream] Error: {e}")
    finally:
        tello_stream.stop()
