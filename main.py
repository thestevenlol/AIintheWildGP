import cv2
import mediapipe as mp
from mediapipe.tasks.python.vision import GestureRecognizerResult


def main():
    # Define MediaPipe components
    BaseOptions = mp.tasks.BaseOptions
    GestureRecognizer = mp.tasks.vision.GestureRecognizer
    GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    
    model_path = 'models/gesture_recognizer.task'
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
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    with GestureRecognizer.create_from_options(options) as recognizer:
        timestamp = 0  # Frame counter for the timestamp
        
        while cap.isOpened():
            # Read a frame from the webcam
            success, frame = cap.read()
            if not success:
                print("Error: Could not read frame from webcam.")
                break
            
            # Convert the frame to RGB (MediaPipe requires RGB input)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Process the frame with MediaPipe Gesture Recognizer
            recognizer.recognize_async(mp_image, timestamp)
            timestamp += 1
            
            # Display gesture information on the frame if available
            if result_dict['gestures']:
                top_gesture = result_dict['gestures'][0][0]
                if top_gesture.category_name == 'Victory':
                    cv2.putText(frame, f"Fuck you: {top_gesture.score:.2f}", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:   
                    cv2.putText(frame, f"{top_gesture.category_name}: {top_gesture.score:.2f}", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow('MediaPipe Gesture Recognition', frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()