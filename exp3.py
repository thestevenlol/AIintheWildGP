import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os
import time

# --- Configuration ---
MODEL_PATH = 'hand_gesture_model_landmarks.keras'
LABELS_PATH = 'landmark_label_classes.npy' # Path to the saved label mapping
WEBCAM_INDEX = 0

# MediaPipe setup (should match data collection settings)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5)

# Display settings
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8
TEXT_COLOR = (0, 255, 0) # Green
RECT_COLOR = (0, 0, 0)   # Black background for text
TEXT_THICKNESS = 2
POSITION = (20, 40)
CONFIDENCE_THRESHOLD = 0.1 # Adjust as needed
# --- End Configuration ---

# --- Helper Function for Landmark Normalization ---
# IMPORTANT: This MUST be identical to the normalization in the data collection script
def normalize_landmarks(landmarks_mp):
    """Normalizes MediaPipe landmarks relative to wrist and scaled."""
    landmarks = landmarks_mp.landmark
    all_coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    relative_coords = all_coords - all_coords[0] # Relative to wrist

    # Normalize scale (using distance between wrist and middle finger MCP - landmark 9)
    wrist_middle_mcp_dist = np.linalg.norm(relative_coords[9])
    if wrist_middle_mcp_dist < 1e-6:
         wrist_middle_mcp_dist = 1 # Prevent division by zero

    normalized_coords = relative_coords / wrist_middle_mcp_dist
    normalized_landmarks_flat = normalized_coords.flatten() # Flatten to 1D array
    return normalized_landmarks_flat

# --- Main Script ---

# 1. Load Model and Labels
if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
    print(f"Error: Model ({MODEL_PATH}) or Labels ({LABELS_PATH}) file not found.")
    exit()

try:
    print("Loading model and labels...")
    model = tf.keras.models.load_model(MODEL_PATH)
    CLASS_NAMES = np.load(LABELS_PATH, allow_pickle=True) # Load the label mapping
    print(f"Model loaded. Found {len(CLASS_NAMES)} classes:", CLASS_NAMES)
except Exception as e:
    print(f"Error loading model or labels: {e}")
    exit()

# 2. Initialize Webcam
print(f"Initializing webcam (index {WEBCAM_INDEX})...")
cap = cv2.VideoCapture(WEBCAM_INDEX)
if not cap.isOpened():
    print(f"Error: Could not open webcam with index {WEBCAM_INDEX}.")
    exit()
print("Webcam initialized.")

prev_time = 0 # For FPS

try:
    while True:
        # 3. Read Frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame. Exiting ...")
            break

        display_frame = frame.copy()

        # 4. Process with MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = hands.process(frame_rgb)
        frame_rgb.flags.writeable = True
        # No need to convert back if only drawing landmarks/text

        prediction_text = "Detecting..." # Default text

        # 5. Extract, Normalize, Predict if Hand Detected
        if results.multi_hand_landmarks:
            # Draw landmarks
            for hand_landmarks in results.multi_hand_landmarks:
                 mp_drawing.draw_landmarks(
                    display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get landmarks for the first hand
            hand_landmarks = results.multi_hand_landmarks[0]

            # Normalize landmarks using the helper function
            normalized_data = normalize_landmarks(hand_landmarks)

            # Prepare data for model prediction (add batch dimension)
            input_data = np.expand_dims(normalized_data, axis=0)

            # Predict
            predictions = model.predict(input_data)
            score = tf.nn.softmax(predictions[0]) # Apply softmax if needed

            predicted_class_index = np.argmax(score)
            confidence = np.max(score)

            # Update text if confidence is high enough
            if confidence >= CONFIDENCE_THRESHOLD:
                predicted_class_name = CLASS_NAMES[predicted_class_index]
                prediction_text = f"{predicted_class_name} ({confidence*100:.1f}%)"
            else:
                prediction_text = f"Detecting... ({confidence*100:.1f}%)" # Show confidence even if low

        # 6. Display Prediction & FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time
        fps_text = f"FPS: {int(fps)}"

        # Add background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(prediction_text, FONT, FONT_SCALE, TEXT_THICKNESS)
        cv2.rectangle(display_frame, (POSITION[0], POSITION[1] - text_height - baseline),
                      (POSITION[0] + text_width, POSITION[1] + baseline), RECT_COLOR, -1)

        # Put prediction text
        cv2.putText(display_frame, prediction_text, POSITION, FONT, FONT_SCALE, TEXT_COLOR, TEXT_THICKNESS)

        # Put FPS text
        cv2.putText(display_frame, fps_text, (display_frame.shape[1] - 150, 40), FONT, FONT_SCALE, TEXT_COLOR, TEXT_THICKNESS)

        # 7. Show Frame
        cv2.imshow('MediaPipe Gesture Recognition (Q to Quit)', display_frame)

        # 8. Quit Key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n'q' pressed, exiting.")
            break

except Exception as e:
    print(f"\nAn error occurred during execution: {e}")
except KeyboardInterrupt:
     print("\nCtrl+C detected, exiting.")

finally:
    # 9. Release Resources
    print("Releasing resources...")
    if 'cap' in locals() and cap.isOpened():
        cap.release()
    if 'hands' in locals():
        hands.close()
    cv2.destroyAllWindows()
    print("Webcam released and windows closed.")