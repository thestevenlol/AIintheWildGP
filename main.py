import cv2
import mediapipe as mp
import numpy as np
import time
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import GestureRecognizerResult
from djitellopy import Tello

# Force OpenCV to use XWayland
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["GDK_BACKEND"] = "x11"

# Try to use display, but continue if not available
DISPLAY_AVAILABLE = True
try:
    test_window = cv2.namedWindow("Test", cv2.WINDOW_NORMAL)
    cv2.destroyWindow("Test")
except:
    DISPLAY_AVAILABLE = False
    print("Warning: Display not available, running in headless mode")

def main():
    # Define MediaPipe components
    BaseOptions = mp.tasks.BaseOptions
    GestureRecognizer = mp.tasks.vision.GestureRecognizer
    GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    
    tello = Tello()
    tello.connect()
    tello.streamon()
    
    model_path = '/home/jack/College/AIintheWildGP/models/gesture_recognizer.task'
    base_options = BaseOptions(model_asset_path=model_path)
    
    # Store latest results
    result_dict = {'gestures': []}
    
    def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
        # Store the result for use in main loop
        result_dict['gestures'] = result.gestures
        if result.gestures:
            top_gesture = result.gestures[0][0]
            print(f'Gesture: {top_gesture.category_name}, Score: {top_gesture.score:.2f}')
    
    # Create a gesture recognizer instance with the live stream mode
    options = GestureRecognizerOptions(
        base_options=base_options,
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=print_result)
    
    # Initialize the webcam
    # cap = cv2.VideoCapture(0)
    # if not cap.isOpened():
    #     print("Error: Could not open webcam.")
    #     return
    
    frame_read = tello.get_frame_read()
    timestamp = 0  # Frame counter for the timestamp
    with GestureRecognizer.create_from_options(options) as recognizer:
        while True:
            # Get the current frame from Tello
            frame = frame_read.frame
            
            if frame is None:
                continue
            
            # Process the frame with MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Process the frame with MediaPipe Gesture Recognizer
            recognizer.recognize_async(mp_image, timestamp)
            timestamp += 1
            
            # Display gesture information on the frame if available
            if result_dict['gestures']:
                top_gesture = result_dict['gestures'][0][0]
                cv2.putText(frame, f"{top_gesture.category_name}: {top_gesture.score:.2f}", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Example: Control drone based on gestures
                # Uncomment these lines to enable gesture control
                # if top_gesture.category_name == 'Thumb_Up' and top_gesture.score > 0.8:
                #     tello.takeoff()
                # elif top_gesture.category_name == 'Thumb_Down' and top_gesture.score > 0.8:
                #     tello.land()
            
            # Modify the display part:
            if DISPLAY_AVAILABLE:
                try:
                    cv2.imshow('Tello MediaPipe Gesture Recognition', frame)
                    # Break the loop if 'q' is pressed
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except:
                    # If display fails, print info instead
                    if 'top_gesture' in locals() and top_gesture:
                        print(f"Frame processed: {top_gesture.category_name}: {top_gesture.score:.2f}")
            else:
                # In headless mode, add a small delay and print status
                time.sleep(0.1)
                if result_dict['gestures']:
                    top_gesture = result_dict['gestures'][0][0]
                    print(f"Frame processed: {top_gesture.category_name}: {top_gesture.score:.2f}")
    
    # Release resources
    cv2.destroyAllWindows()
    tello.streamoff()
    tello.end()


if __name__ == "__main__":
    main()