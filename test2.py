import cv2
import numpy as np
import math
import time
from enum import Enum
from djitellopy import Tello

# --- States ---
class DroneState(Enum):
    IDLE = 0           # On the ground, disconnected or connected but not taken off
    CONNECTED = 1      # Connected, on the ground
    HOVERING = 2       # In the air, ready
    SCANNING = 3       # In the air, performing scan
    PATH_STORED = 4    # Scan complete, path stored, ready to execute
    EXECUTING = 5      # Flying the stored path
    EXECUTION_DONE = 6 # Path finished, hovering
    LANDING = 7        # Landing sequence initiated

# --- Constants ---
# REMOVED: LOWER_GREEN, UPPER_GREEN

# Grayscale Threshold for Black Detection
# Pixels with grayscale value BELOW this will be considered black (and turned white in the mask)
BLACK_THRESHOLD_VALUE = 40 # <<<--- ADJUST THIS VALUE (Lower = stricter black, Higher = includes darker grays)

# Morphological Operations
KERNEL_SIZE = 2
MORPH_ITERATIONS = 1 # Increase if mask is too noisy (try 2 or 3)
MIN_CONTOUR_AREA = 100 # Pixels

# --- Path Following Parameters ---
CONTOUR_SIMPLIFICATION_FACTOR = 0.008 # Epsilon for approxPolyDP

# --- CRITICAL ASSUMPTION: Scale Factor ---
PIXELS_TO_METERS_SCALE = 0.001 # <--- !!! MUST CALIBRATE THIS VALUE !!!

# --- Tello Control Parameters ---
TELLO_SPEED = 30  # Movement speed (cm/s) - START SLOW
WAIT_TIME_BUFFER = 0.5 # Extra seconds to wait

# --- Global Variables ---
tello = None
state = DroneState.IDLE
stored_path_pixels = []
current_waypoint_index = 0
frame_reader = None
last_frame = None

# --- Tello Functions (connect_tello, safe_takeoff, safe_land, cleanup - remain the same) ---
def connect_tello():
    global tello, state, frame_reader
    try:
        tello = Tello()
        tello.connect()
        print("Tello connected.")
        print(f"Battery: {tello.get_battery()}%")
        if tello.get_battery() < 20:
            print("WARNING: Battery low! Landing recommended.")
        tello.streamoff() # Ensure stream is off first
        tello.streamon()
        frame_reader = tello.get_frame_read()
        time.sleep(1) # Allow stream to initialize
        state = DroneState.CONNECTED
        return True
    except Exception as e:
        print(f"Failed to connect to Tello: {e}")
        tello = None
        state = DroneState.IDLE
        return False

def safe_takeoff():
    global state
    if tello and state == DroneState.CONNECTED:
        try:
            print("Taking off...")
            tello.takeoff()
            time.sleep(2) # Allow takeoff to stabilize
            state = DroneState.HOVERING
            print("Hovering.")
            return True
        except Exception as e:
            print(f"Takeoff failed: {e}")
            try: tello.land()
            except: pass
            state = DroneState.CONNECTED # Remain connected but on ground
            return False
    return False

def safe_land():
    global state
    if tello and (state != DroneState.IDLE and state != DroneState.CONNECTED and state != DroneState.LANDING):
        try:
            state = DroneState.LANDING
            print("Landing...")
            tello.land()
            time.sleep(3) # Allow landing
            state = DroneState.CONNECTED # Back to connected state on ground
            print("Landed.")
            return True
        except Exception as e:
            print(f"Landing failed: {e}")
            state = DroneState.CONNECTED
            return False
    return False

def cleanup():
    global tello, state
    print("Cleaning up...")
    if tello:
        try:
            if state not in [DroneState.IDLE, DroneState.CONNECTED, DroneState.LANDING]:
                print("Auto-landing before exit...")
                safe_land()
            tello.streamoff()
        except Exception as e:
            print(f"Error during Tello cleanup: {e}")
    cv2.destroyAllWindows()
    print("Cleanup complete.")

# --- Path Execution Function (execute_path_step - remains the same) ---
def execute_path_step():
    global current_waypoint_index, state
    if not stored_path_pixels or current_waypoint_index >= len(stored_path_pixels):
        print("Execution complete or no path.")
        state = DroneState.EXECUTION_DONE
        return

    target_px, target_py = stored_path_pixels[current_waypoint_index]
    if current_waypoint_index == 0:
        print(f"Skipping move to first waypoint {current_waypoint_index}: ({target_px}, {target_py}) for now.")
        current_waypoint_index += 1
        if current_waypoint_index >= len(stored_path_pixels):
            state = DroneState.EXECUTION_DONE
        return
    else:
        prev_px, prev_py = stored_path_pixels[current_waypoint_index - 1]

    delta_px = target_px - prev_px
    delta_py = target_py - prev_py

    dx_meters = delta_px * PIXELS_TO_METERS_SCALE
    dy_meters = -delta_py * PIXELS_TO_METERS_SCALE

    tello_x_cm = 0
    tello_y_cm = -dx_meters * 100
    tello_z_cm = dy_meters * 100

    tello_y_cm = int(np.clip(tello_y_cm, -500, 500))
    tello_z_cm = int(np.clip(tello_z_cm, -500, 500))

    if abs(tello_y_cm) < 5 and abs(tello_z_cm) < 5:
         print(f"Skipping tiny move for waypoint {current_waypoint_index}.")
         current_waypoint_index += 1
         return

    distance_cm = math.sqrt(tello_y_cm**2 + tello_z_cm**2)
    wait_time = (distance_cm / TELLO_SPEED) + WAIT_TIME_BUFFER

    print(f"Executing step to waypoint {current_waypoint_index}:")
    print(f"  Delta Pixels: ({delta_px}, {delta_py})")
    print(f"  Delta Meters: ({dx_meters:.3f}, {dy_meters:.3f})")
    print(f"  Tello Cmd (X,Y,Z cm): ({tello_x_cm}, {tello_y_cm}, {tello_z_cm})")
    print(f"  Est. Time: {wait_time:.2f}s")

    try:
        tello.go_xyz_speed(tello_x_cm, tello_y_cm, tello_z_cm, TELLO_SPEED)
        time.sleep(wait_time)
        current_waypoint_index += 1
        if current_waypoint_index >= len(stored_path_pixels):
            print("Execution finished.")
            state = DroneState.EXECUTION_DONE
    except Exception as e:
        print(f"Error during Tello movement: {e}")
        print("Aborting execution and attempting to hover.")
        try:
            tello.send_rc_control(0,0,0,0)
        except: pass
        state = DroneState.HOVERING

# --- Main Loop ---
if connect_tello():
    try:
        cv2.namedWindow("Tello Path Scanner")
        cv2.namedWindow("Black Mask") # <-- Renamed window

        while True:
            if not tello:
                print("Tello connection lost.")
                break

            frame = frame_reader.frame
            if frame is None:
                print("No frame received")
                time.sleep(0.1)
                continue

            frame_height, frame_width = frame.shape[:2]
            display_frame = frame.copy()

            # --- Image Processing for Black Detection ---
            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Thresholding: Pixels below BLACK_THRESHOLD_VALUE become white (255), others black (0)
            ret, initial_mask = cv2.threshold(gray_image, BLACK_THRESHOLD_VALUE, 255, cv2.THRESH_BINARY_INV)

            # Morphological Closing to clean up mask
            kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
            refined_mask = cv2.morphologyEx(initial_mask, cv2.MORPH_CLOSE, kernel, iterations=MORPH_ITERATIONS)
            # Optional: Add Opening first if you have small black noise spots you want removed
            # refined_mask = cv2.morphologyEx(initial_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            # refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel, iterations=MORPH_ITERATIONS)

            # --- State Machine Logic & Display (Mostly unchanged) ---
            status_text = f"State: {state.name} | Bat: {tello.get_battery()}%"
            controls_text = ""

            if state == DroneState.IDLE:
                status_text = "State: IDLE (Not Connected?)"
                controls_text = "Press 'c' to Connect"
            # ... (other states remain the same, displaying status and controls) ...
            elif state == DroneState.CONNECTED:
                controls_text = "Press 't' to Takeoff"
            elif state == DroneState.HOVERING:
                controls_text = "Press 's' to Scan Path | 'l' to Land"
            elif state == DroneState.SCANNING:
                status_text += " (Scanning...)"
                # --- Scanning Logic (Uses refined_mask from black detection) ---
                contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Use refined_mask
                largest_contour = None
                max_area = 0
                if contours:
                    contours = sorted(contours, key=cv2.contourArea, reverse=True)
                    if cv2.contourArea(contours[0]) > MIN_CONTOUR_AREA:
                         largest_contour = contours[0]

                if largest_contour is not None:
                    perimeter = cv2.arcLength(largest_contour, True)
                    epsilon = CONTOUR_SIMPLIFICATION_FACTOR * perimeter
                    simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
                    stored_path_pixels = [tuple(pt[0]) for pt in simplified_contour]

                    if len(stored_path_pixels) > 1:
                        print(f"Path Scanned: {len(stored_path_pixels)} waypoints.")
                        if stored_path_pixels[0][0] > stored_path_pixels[-1][0]:
                            print("Path seems right-to-left, reversing for execution.")
                            stored_path_pixels.reverse()
                        state = DroneState.PATH_STORED
                    else:
                        print("Scan failed: Not enough points after simplification.")
                        state = DroneState.HOVERING
                else:
                    print("Scan failed: No suitable contour found.")
                    state = DroneState.HOVERING

            elif state == DroneState.PATH_STORED:
                controls_text = "Press 'g' to Go | 'r' to Rescan | 'l' to Land"
                if len(stored_path_pixels) > 1:
                    path_np = np.array(stored_path_pixels)
                    cv2.polylines(display_frame, [path_np], isClosed=False, color=(0, 255, 0), thickness=3)
                    for i, pt in enumerate(stored_path_pixels):
                         cv2.circle(display_frame, pt, 4, (0, 0, 255), -1)

            elif state == DroneState.EXECUTING:
                 status_text += f" (Waypoint {current_waypoint_index}/{len(stored_path_pixels)})"
                 controls_text = "Executing... Press 'l' for EMERGENCY LAND"
                 if len(stored_path_pixels) > 1:
                     path_np = np.array(stored_path_pixels)
                     cv2.polylines(display_frame, [path_np], isClosed=False, color=(0, 165, 255), thickness=2)
                     if current_waypoint_index < len(stored_path_pixels):
                         target_pixel_pt = stored_path_pixels[current_waypoint_index]
                         cv2.circle(display_frame, target_pixel_pt, 8, (0, 0, 255), -1)
                 execute_path_step()

            elif state == DroneState.EXECUTION_DONE:
                controls_text = "Press 'r' to Reset (Hover) | 'l' to Land"
                if len(stored_path_pixels) > 1:
                    path_np = np.array(stored_path_pixels)
                    cv2.polylines(display_frame, [path_np], isClosed=False, color=(0, 255, 0), thickness=3)
            elif state == DroneState.LANDING:
                 status_text = "State: LANDING..."


            # --- Display Info Text ---
            cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(display_frame, controls_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

            # --- Show Windows ---
            cv2.imshow("Tello Path Scanner", display_frame)
            cv2.imshow("Black Mask", refined_mask) # <-- Show the black mask

            # --- Keyboard Controls (Remain the same) ---
            k = cv2.waitKey(1) & 0xFF
            if k == 27: # ESC
                print("ESC pressed. Landing and exiting.")
                break
            elif k == ord('c') and state == DroneState.IDLE:
                 connect_tello()
            elif k == ord('t') and state == DroneState.CONNECTED:
                 safe_takeoff()
            elif k == ord('l'): # Land
                 print("Landing initiated by keypress.")
                 safe_land()
            elif k == ord('s') and state == DroneState.HOVERING:
                 print("Starting Scan...")
                 state = DroneState.SCANNING
            elif k == ord('g') and state == DroneState.PATH_STORED:
                 print("Starting Path Execution...")
                 current_waypoint_index = 0
                 state = DroneState.EXECUTING
            elif k == ord('r') and (state == DroneState.PATH_STORED or state == DroneState.EXECUTION_DONE):
                 print("Resetting path and returning to Hovering.")
                 stored_path_pixels = []
                 current_waypoint_index = 0
                 state = DroneState.HOVERING

    except Exception as e:
        print(f"An error occurred in the main loop: {e}")
    finally:
        cleanup()
else:
    print("Could not connect to Tello. Exiting.")