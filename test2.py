import cv2
import numpy as np
import math
from enum import Enum # For states
from djitellopy import Tello

# --- States ---
class DroneState(Enum):
    IDLE = 0
    SCANNING = 1
    PATH_STORED = 2
    EXECUTING = 3
    EXECUTION_DONE = 4

# --- Constants ---
LOWER_GREEN = np.array([35, 50, 50])
UPPER_GREEN = np.array([85, 255, 255])
KERNEL_SIZE = 3
MORPH_ITERATIONS = 1
MIN_CONTOUR_AREA = 100

# --- Path Following Parameters ---
# How much to simplify the contour. Smaller value = more points, closer fit.
# Value is epsilon factor for approxPolyDP (percentage of arc length)
CONTOUR_SIMPLIFICATION_FACTOR = 0.007 # Try values between 0.005 and 0.02

# --- CRITICAL ASSUMPTION: Scale Factor ---
# How many meters in the real world correspond to one pixel?
# This MUST be calibrated based on drone distance and camera FOV.
# Example: If drone is 1m away and 100 pixels span 0.2m horizontally,
# scale = 0.2 / 100 = 0.002 meters/pixel
PIXELS_TO_METERS_SCALE = 0.002 # <--- !!! ADJUST THIS VALUE (Calibration Needed) !!!

# --- Drone Simulation ---
stored_path_pixels = []
stored_path_meters_relative = []
current_waypoint_index = 0

tello = Tello()
tello.connect()
tello.streamon()

def simulate_drone_goto(target_meters_relative):
    """Simulates telling the drone to move to a relative position."""
    # Assumes target is relative to where the drone was when execution started.
    # Assumes Drone Frame: +X is Right, +Y is UP, +Z is Forward (away from wall)
    # We only care about X and Y for wall tracing.
    print(f"SIM DRONE: GOTO Relative Waypoint (X, Y): ({target_meters_relative[0]:.3f}, {target_meters_relative[1]:.3f}) meters")
    # In a real system:
    # drone.goto_position_relative(target_meters_relative[0], target_meters_relative[1], 0, speed=...)
    # wait_until_drone_reaches_target()

cv2.namedWindow("Path Scanner/Executor")
cv2.namedWindow("Green Mask")

state = DroneState.IDLE
frame_center_when_scanned = None # Store center point during scan

# --- Processing Loop ---
while True:
    frame = tello.get_frame_read().frame
    if frame is None:
        print("Failed to grab frame")
        break

    frame_height, frame_width = frame.shape[:2]
    frame_center_x = frame_width // 2
    frame_center_y = frame_height // 2

    # --- Image Processing ---
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    initial_mask = cv2.inRange(hsv_image, LOWER_GREEN, UPPER_GREEN)
    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
    refined_mask = cv2.morphologyEx(initial_mask, cv2.MORPH_CLOSE, kernel, iterations=MORPH_ITERATIONS)

    # Make a copy for drawing without affecting calculations
    display_frame = frame.copy()

    # --- State Machine Logic ---
    if state == DroneState.IDLE:
        cv2.putText(display_frame, "State: IDLE. Press 's' to Scan.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    elif state == DroneState.SCANNING:
        cv2.putText(display_frame, "State: SCANNING...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        contours, hierarchy = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        largest_contour = None
        max_area = 0
        if contours:
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > MIN_CONTOUR_AREA and area > max_area:
                    max_area = area
                    largest_contour = contour

        if largest_contour is not None:
            # Simplify the contour
            perimeter = cv2.arcLength(largest_contour, True)
            epsilon = CONTOUR_SIMPLIFICATION_FACTOR * perimeter
            simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

            # Store the path (pixel coordinates)
            # Ensure it's a simple list of (x, y) tuples
            stored_path_pixels = [tuple(pt[0]) for pt in simplified_contour]

            if len(stored_path_pixels) > 1:
                 # Store the frame center at the time of scanning for relative calculations
                frame_center_when_scanned = (frame_center_x, frame_center_y)
                print(f"Path Scanned: {len(stored_path_pixels)} waypoints found.")
                state = DroneState.PATH_STORED
            else:
                print("Scan failed: Not enough points in simplified contour.")
                state = DroneState.IDLE # Go back to idle if scan fails
        else:
            print("Scan failed: No suitable contour found.")
            state = DroneState.IDLE # Go back to idle if scan fails


    elif state == DroneState.PATH_STORED:
        cv2.putText(display_frame, f"State: PATH STORED ({len(stored_path_pixels)} pts). Press 'g' to Go.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # Draw the stored path
        if len(stored_path_pixels) > 1:
            cv2.polylines(display_frame, [np.array(stored_path_pixels)], isClosed=False, color=(0, 255, 0), thickness=3)
            for i, pt in enumerate(stored_path_pixels):
                 cv2.circle(display_frame, pt, 4, (0, 0, 255), -1)
                 cv2.putText(display_frame, str(i), (pt[0]+5, pt[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)


    elif state == DroneState.EXECUTING:
        cv2.putText(display_frame, f"State: EXECUTING Waypoint {current_waypoint_index}/{len(stored_path_meters_relative)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Draw the full target path
        if len(stored_path_pixels) > 1:
             cv2.polylines(display_frame, [np.array(stored_path_pixels)], isClosed=False, color=(0, 165, 255), thickness=2) # Orange full path

        # Highlight current target waypoint
        if current_waypoint_index < len(stored_path_pixels):
            target_pixel_pt = stored_path_pixels[current_waypoint_index]
            cv2.circle(display_frame, target_pixel_pt, 8, (0, 0, 255), -1) # Highlight target


        # --- SIMULATE DRONE ACTION ---
        if current_waypoint_index < len(stored_path_meters_relative):
            target_meters = stored_path_meters_relative[current_waypoint_index]
            simulate_drone_goto(target_meters)

            # In reality, you'd wait for confirmation here. We just move to the next.
            current_waypoint_index += 1
        else:
            print("EXECUTION COMPLETE.")
            state = DroneState.EXECUTION_DONE

    elif state == DroneState.EXECUTION_DONE:
         cv2.putText(display_frame, "State: EXECUTION DONE. Press 'r' to Reset.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
         # Keep drawing the path
         if len(stored_path_pixels) > 1:
            cv2.polylines(display_frame, [np.array(stored_path_pixels)], isClosed=False, color=(0, 255, 0), thickness=3)


    # --- Display ---
    cv2.imshow("Green Mask", refined_mask)
    cv2.imshow("Path Scanner/Executor", display_frame)

    # --- Keyboard Controls ---
    k = cv2.waitKey(1) & 0xFF
    if k == 27: # ESC
        print("Escape hit, closing...")
        break
    elif k == ord('s') and state == DroneState.IDLE:
        print("Starting Scan...")
        state = DroneState.SCANNING
    elif k == ord('g') and state == DroneState.PATH_STORED:
        print("Starting Execution...")
        # --- Convert stored pixel path to relative meters ---
        stored_path_meters_relative = []
        if stored_path_pixels and frame_center_when_scanned:
            scan_center_x, scan_center_y = frame_center_when_scanned
            # Optional: Find the top-leftmost point to start consistently
            # start_index = np.argmin([p[0] + p[1] for p in stored_path_pixels]) # Heuristic for top-left
            # path_to_process = stored_path_pixels[start_index:] + stored_path_pixels[:start_index] # Reorder
            path_to_process = stored_path_pixels # Keep original order for now

            # Reference point (first point in path) in pixel coords relative to scan center
            ref_px, ref_py = path_to_process[0]
            delta_ref_px = ref_px - scan_center_x
            delta_ref_py = ref_py - scan_center_y

            for i, (px, py) in enumerate(path_to_process):
                delta_px = px - scan_center_x
                delta_py = py - scan_center_y

                # Calculate target meters relative to scan center
                target_x_scan_relative = delta_px * PIXELS_TO_METERS_SCALE
                target_y_scan_relative = -delta_py * PIXELS_TO_METERS_SCALE # Y inverted

                # If it's the first point, the relative move is just its position
                if i == 0:
                    relative_move_x = target_x_scan_relative
                    relative_move_y = target_y_scan_relative
                else:
                    # Calculate previous point's position relative to scan center
                    prev_px, prev_py = path_to_process[i-1]
                    delta_prev_px = prev_px - scan_center_x
                    delta_prev_py = prev_py - scan_center_y
                    prev_x_scan_relative = delta_prev_px * PIXELS_TO_METERS_SCALE
                    prev_y_scan_relative = -delta_prev_py * PIXELS_TO_METERS_SCALE

                    # The relative move is the difference from the previous waypoint
                    # This might be better for some drone APIs (move BY dx, dy)
                    # relative_move_x = target_x_scan_relative - prev_x_scan_relative
                    # relative_move_y = target_y_scan_relative - prev_y_scan_relative

                    # Alternative: Calculate absolute target relative to drone's *starting* position (where scan began)
                    relative_move_x = target_x_scan_relative
                    relative_move_y = target_y_scan_relative


                stored_path_meters_relative.append((relative_move_x, relative_move_y))

            current_waypoint_index = 0
            state = DroneState.EXECUTING
        else:
            print("Error: No path stored or scan center missing.")

    elif k == ord('r'): # Reset
        print("Resetting state.")
        stored_path_pixels = []
        stored_path_meters_relative = []
        current_waypoint_index = 0
        frame_center_when_scanned = None
        state = DroneState.IDLE


# --- Cleanup ---
tello.streamoff()
cv2.destroyAllWindows()