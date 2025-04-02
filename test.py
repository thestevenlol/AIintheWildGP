import cv2
import numpy as np

# --- Constants (Adjust these!) ---

# HSV Green Range (Tune for your specific green marker and lighting)
LOWER_GREEN = np.array([35, 50, 50])
UPPER_GREEN = np.array([85, 255, 255])

# Morphological Kernel Size
KERNEL_SIZE = 3

# Morphological Iterations
MORPH_ITERATIONS = 1

# Minimum Contour Area (Filter out small noise) - ADJUST THIS!
MIN_CONTOUR_AREA = 100 # Pixels

# Center Deadzones (How close to center counts as "centered"?)
# Expressed as a fraction of the frame dimension
HORIZONTAL_CENTER_DEADZONE_FRACTION = 0.15 # +/- 15% of width from center (Turning)
VERTICAL_CENTER_DEADZONE_FRACTION = 0.15   # +/- 15% of height from center (Up/Down)

# --- Drone Command Placeholder (Prints to Console) ---
def send_drone_command(command):
    """Prints the intended drone command to the console."""
    print(f"COMMAND: {command}")
    # In a real application, you would send commands here
    # e.g., drone.move_up(speed=...), drone.turn_left(angle=...), drone.hover()

# --- Webcam Setup ---
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Error: Could not open webcam.")
    exit()

cv2.namedWindow("Line Following View (Wall)")
cv2.namedWindow("Green Mask (Refined)")

# --- Processing Loop ---
while True:
    ret, frame = cam.read()
    if not ret:
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
    # Optional: Add MORPH_OPEN to remove small noise before closing
    # refined_mask = cv2.morphologyEx(initial_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    # refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel, iterations=MORPH_ITERATIONS)


    # --- Contour Detection and Filtering ---
    contours, hierarchy = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = None
    max_area = 0
    line_found = False
    cx, cy = 0, 0 # Centroid coordinates

    if contours:
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > MIN_CONTOUR_AREA and area > max_area:
                max_area = area
                largest_contour = contour

    # --- Centroid Calculation and Control Logic ---
    command = "HOVER" # Default command if no line or centered

    if largest_contour is not None:
        line_found = True
        # Calculate centroid
        M = cv2.moments(largest_contour)
        if M["m00"] != 0: # Avoid division by zero
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Draw the largest contour and its centroid
            cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 3)
            cv2.circle(frame, (cx, cy), 7, (0, 0, 255), -1) # Red circle at centroid

            # --- Determine Horizontal Position (Turning) ---
            h_deadzone_width_pixels = int(HORIZONTAL_CENTER_DEADZONE_FRACTION * frame_width / 2)
            left_bound = frame_center_x - h_deadzone_width_pixels
            right_bound = frame_center_x + h_deadzone_width_pixels

            # --- Determine Vertical Position (Up/Down) ---
            v_deadzone_height_pixels = int(VERTICAL_CENTER_DEADZONE_FRACTION * frame_height / 2)
            top_bound = frame_center_y - v_deadzone_height_pixels   # Lower Y value is higher up
            bottom_bound = frame_center_y + v_deadzone_height_pixels # Higher Y value is lower down

            # Draw deadzone lines for visualization
            # Horizontal (for turning)
            cv2.line(frame, (left_bound, 0), (left_bound, frame_height), (255, 255, 0), 1) # Cyan
            cv2.line(frame, (right_bound, 0), (right_bound, frame_height), (255, 255, 0), 1) # Cyan
            cv2.line(frame, (frame_center_x, 0), (frame_center_x, frame_height), (0, 0, 255), 1) # Red center X
            # Vertical (for up/down)
            cv2.line(frame, (0, top_bound), (frame_width, top_bound), (0, 255, 255), 1) # Yellow
            cv2.line(frame, (0, bottom_bound), (frame_width, bottom_bound), (0, 255, 255), 1) # Yellow
            cv2.line(frame, (0, frame_center_y), (frame_width, frame_center_y), (0, 0, 255), 1) # Red center Y

            # --- Command Logic: Prioritize Turning ---
            if cx < left_bound:
                command = "LEFT"
            elif cx > right_bound:
                command = "RIGHT"
            else:
                # --- If Horizontally Centered, Check Vertical Position ---
                if cy < top_bound:
                    command = "UP" # Line is above center
                elif cy > bottom_bound:
                    command = "DOWN" # Line is below center
                else:
                    command = "HOVER" # Line is centered both ways (or close enough)

        else:
             # Centroid calculation failed
             line_found = False
             command = "HOVER" # Hover if calculation fails

    else:
        # No significant contour found
        line_found = False
        command = "SEARCH" # Or HOVER - decide behavior when line is lost

    # --- Send Command to Console ---
    send_drone_command(command)

    # --- Display Information ---
    cv2.putText(frame, f"Detected Command: {command}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    if line_found:
        cv2.putText(frame, f"Centroid: ({cx}, {cy})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
    else:
         cv2.putText(frame, "Line not found", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)


    cv2.imshow("Green Mask (Refined)", refined_mask)
    cv2.imshow("Line Following View (Wall)", frame)

    # --- Exit Condition (Press ESC) ---
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        print("Escape hit, closing...")
        break

# --- Cleanup ---
send_drone_command("HOVER") # Ensure hover command is sent before exiting
cam.release()
cv2.destroyAllWindows()