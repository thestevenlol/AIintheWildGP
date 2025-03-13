import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["GDK_BACKEND"] = "x11"

import cv2
import time
from djitellopy import Tello

def main():
    # Initialize Tello
    tello = Tello()
    tello.connect()
    print(f"Battery: {tello.get_battery()}%")
    
    # Start video stream
    tello.streamon()
    frame_read = tello.get_frame_read()
    
    # Create window explicitly before showing frames
    cv2.namedWindow("Tello Stream", cv2.WINDOW_NORMAL)
    time.sleep(0.5)  # Small delay to ensure window creation
    
    try:
        while True:
            # Get frame
            frame = frame_read.frame
            if frame is None:
                print("Received empty frame, continuing...")
                time.sleep(0.1)
                continue
            
            # Process frame - resize for stability
            try:
                frame = cv2.resize(frame, (640, 480))
                # Display
                cv2.imshow("Tello Stream", frame)
                print("Frame displayed", end="\r")  # Print feedback but overwrite each line
            except Exception as e:
                print(f"Error processing frame: {e}")
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Clean up
        cv2.destroyAllWindows()
        tello.streamoff()
        tello.end()

if __name__ == "__main__":
    main()