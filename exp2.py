import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import time
import keyboard # Use 'pip install keyboard'

# --- Configuration ---
GESTURE_NAME = "land"  # <<< CHANGE THIS for each gesture ('up', 'down', 'left', etc.)
CAPTURE_INTERVAL = 0.02 # <<< Interval in seconds (faster capture often okay for landmarks)
OUTPUT_CSV_FILE = "gesture_landmark_data.csv" # All data saved here
WEBCAM_INDEX = 0
# --- End Configuration ---

# MediaPipe Hands setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,       # Process video stream
    max_num_hands=1,               # Detect only one hand for simplicity
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5)

# CSV Header (if file doesn't exist)
header = ['label']
# Add 21 landmarks * 3 coordinates (x, y, z) = 63 features
for i in range(21):
    header += [f'x{i}', f'y{i}', f'z{i}']

if not os.path.exists(OUTPUT_CSV_FILE):
    with open(OUTPUT_CSV_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
    print(f"Created CSV file: {OUTPUT_CSV_FILE}")
else:
    print(f"Appending data to existing CSV file: {OUTPUT_CSV_FILE}")


# Initialize webcam
cap = cv2.VideoCapture(WEBCAM_INDEX)
if not cap.isOpened():
    print(f"Error: Could not open webcam with index {WEBCAM_INDEX}.")
    exit()

print("\n--- MediaPipe Hand Landmark Collection ---")
print(f"Gesture to capture: {GESTURE_NAME}")
print(f"Capture interval: {CAPTURE_INTERVAL} seconds")
print("Hold SPACEBAR to start/continue capturing landmarks.")
print("Press 'q' in the webcam window to quit.")
print("--------------------------------------\n")

last_capture_time = 0
sample_counter = 0
capturing = False

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame. Exiting...")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False # Improve performance
        results = hands.process(frame_rgb)
        frame_rgb.flags.writeable = True
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR) # Back to BGR for display

        current_time = time.time()
        is_space_pressed = keyboard.is_pressed('space')

        landmarks_extracted = False
        normalized_landmarks_flat = []

        # --- Landmark Extraction and Normalization ---
        if results.multi_hand_landmarks:
            # Draw landmarks on the frame
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Process only the first detected hand
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = hand_landmarks.landmark

            # --- Normalization ---
            # 1. Get all coordinates
            all_coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

            # 2. Make relative to wrist (landmark 0)
            relative_coords = all_coords - all_coords[0]

            # 3. Normalize scale (using distance between wrist and middle finger MCP - landmark 9)
            # Avoid division by zero if landmarks are too close
            wrist_middle_mcp_dist = np.linalg.norm(relative_coords[9]) # Magni tude of vector from wrist to landmark 9
            if wrist_middle_mcp_dist < 1e-6:
                 wrist_middle_mcp_dist = 1 # Prevent division by zero, use 1 (no scaling)
                 print("Warning: Landmarks too close, potential normalization issue.")


            normalized_coords = relative_coords / wrist_middle_mcp_dist

            # 4. Flatten the normalized coordinates (21 landmarks * 3 coords = 63 features)
            normalized_landmarks_flat = normalized_coords.flatten().tolist()
            landmarks_extracted = True
            # --- End Normalization ---


        # --- Capture Logic ---
        if is_space_pressed:
            if not capturing:
                print("Capturing started...")
                capturing = True

            # Check interval and if landmarks were successfully extracted
            if landmarks_extracted and (current_time - last_capture_time >= CAPTURE_INTERVAL):
                # Save the normalized landmarks + label to CSV
                row_data = [GESTURE_NAME] + normalized_landmarks_flat
                with open(OUTPUT_CSV_FILE, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row_data)

                sample_counter += 1
                print(f"Captured sample {sample_counter} for '{GESTURE_NAME}'")
                last_capture_time = current_time

             # Visual indicator
            cv2.putText(frame, "RECORDING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.circle(frame, (frame.shape[1] - 30, 30), 10, (0, 0, 255), -1)

        else:
            if capturing:
                print("Capturing paused...")
                capturing = False
            cv2.putText(frame, "PAUSED (Hold SPACE)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if not results.multi_hand_landmarks:
                 cv2.putText(frame, "No Hand Detected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


        # Display the frame
        cv2.imshow('MediaPipe Hand Landmarks - Data Collection (Q to Quit)', frame)

        # Quit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n'q' pressed, exiting.")
            break

except Exception as e:
    print(f"\nAn error occurred: {e}")
except KeyboardInterrupt:
    print("\nCtrl+C detected, exiting.")

finally:
    # Release resources
    print("\nReleasing resources...")
    if 'cap' in locals() and cap.isOpened():
        cap.release()
    if 'hands' in locals():
        hands.close()
    cv2.destroyAllWindows()
    print(f"Data saved to: {OUTPUT_CSV_FILE}")
    print(f"Total samples collected in this session for '{GESTURE_NAME}': {sample_counter}")