import cv2
import numpy as np
import math # For atan2
from djitellopy import Tello 

# --- Constants (Adjust these!) ---

# Grayscale Threshold (Lower value detects darker pixels as "black")
# Tune this based on your lighting and how black the line is vs the background.
# Values typically range from 50 to 150. Start around 100.
GRAYSCALE_THRESHOLD = 40 # Pixels below this value are considered black

# Morphological Kernel Size
KERNEL_SIZE = 1 # Often needs to be slightly larger for thresholded images

# Morphological Iterations
MORPH_ITERATIONS = 2 # Might need more iterations to clean up threshold noise

# Minimum Contour Area (Filter out small noise) - ADJUST THIS!
MIN_CONTOUR_AREA = 150 # Pixels (adjust based on line thickness and distance)

# --- Control Parameters ---
# Horizontal Target Offset: Keep centroid slightly left of center (fraction of width)
HORIZONTAL_TARGET_OFFSET_FRACTION = 0.0 # Target center for simplicity first
# Horizontal Deadzone: How far from target before correcting (fraction of width)
HORIZONTAL_DEADZONE_FRACTION = 0.15    # Correct if centroid > +/- 15% from target X

# Vertical Slope Threshold: dy/dx threshold to trigger UP/DOWN
# Steeper than this slope -> move UP/DOWN. Flatter -> HOVER vertically.
# Adjust based on how sharp curves you expect. Higher value = less sensitive.
VERTICAL_SLOPE_THRESHOLD = 0.3 # tan(angle) approx. 17 degrees

# Minimum Horizontal Span (dx) for Slope Calculation
# Don't trust slope if line segment is too vertical or short horizontally
MIN_DX_FOR_SLOPE = 20 # Pixels

# --- Drone Command Placeholder (Prints to Console) ---
def send_drone_command(command):
    """Prints the intended drone command to the console."""
    print(f"COMMAND: {command}")
    # In a real application, you would send commands here

tello = Tello()
tello.connect()
tello.streamon()

cv2.namedWindow("Line Following View (Black Line)")
cv2.namedWindow("Black Line Mask (Refined)")

# --- Processing Loop ---
while True:
    frame = tello.get_frame_read().frame
    if not frame:
        print("Failed to grab frame")
        break

    frame_height, frame_width = frame.shape[:2]
    frame_center_x = frame_width // 2
    frame_center_y = frame_height // 2

    # Calculate target X based on offset
    target_cx = frame_center_x + int(HORIZONTAL_TARGET_OFFSET_FRACTION * frame_width)

    # --- Image Processing for Black Line ---
    # 1. Convert to Grayscale
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. Apply Binary Threshold (Inverted)
    # Pixels below GRAYSCALE_THRESHOLD become white (255), others become black (0)
    # This makes the black line white in the mask, which findContours expects.
    ret, thresh_mask = cv2.threshold(gray_image, GRAYSCALE_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

    # 3. Morphological Operations (Closing) - To fill gaps in the line and remove noise
    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
    # Closing = Dilation followed by Erosion
    refined_mask = cv2.morphologyEx(thresh_mask, cv2.MORPH_CLOSE, kernel, iterations=MORPH_ITERATIONS)
    # Optional: Add Opening (Erosion then Dilation) to remove small white noise specks (background noise)
    # refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel, iterations=1) # Uncomment if needed

    # --- Contour Detection and Filtering ---
    # Find contours on the refined mask (where the line should be white)
    contours, hierarchy = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = None
    max_area = 0
    line_found = False
    cx, cy = 0, 0 # Centroid coordinates
    leftmost_pt = None
    rightmost_pt = None
    line_angle_deg = 0

    if contours:
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > MIN_CONTOUR_AREA and area > max_area:
                max_area = area
                largest_contour = contour

    # --- Analysis if Line Found ---
    command_h = "HOVER_H" # Horizontal component (LEFT, RIGHT, HOVER_H)
    command_v = "HOVER_V" # Vertical component (UP, DOWN, HOVER_V)
    final_command = "SEARCH" # Default if no line

    if largest_contour is not None:
        line_found = True
        final_command = "HOVER" # Default if line is found but centered/flat

        # Calculate centroid
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Find leftmost and rightmost points of the contour
            leftmost_pt = tuple(largest_contour[largest_contour[:, :, 0].argmin()][0])
            rightmost_pt = tuple(largest_contour[largest_contour[:, :, 0].argmax()][0])

            # Draw contour, centroid, and end points on the original frame
            cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2) # Green contour
            cv2.circle(frame, (cx, cy), 7, (0, 0, 255), -1) # Red centroid
            cv2.circle(frame, leftmost_pt, 5, (255, 0, 0), -1) # Blue left
            cv2.circle(frame, rightmost_pt, 5, (0, 255, 255), -1) # Yellow right
            if leftmost_pt and rightmost_pt:
                 cv2.line(frame, leftmost_pt, rightmost_pt, (255, 0, 255), 1) # Magenta line segment

            # --- Determine Horizontal Command (Turning) ---
            h_deadzone_pixels = int(HORIZONTAL_DEADZONE_FRACTION * frame_width / 2)
            left_bound = target_cx - h_deadzone_pixels
            right_bound = target_cx + h_deadzone_pixels

            if cx < left_bound:
                command_h = "LEFT"
            elif cx > right_bound:
                command_h = "RIGHT"
            else:
                command_h = "HOVER_H" # Horizontally within deadzone of target

            # --- Determine Vertical Command (Up/Down based on slope) ---
            dx = rightmost_pt[0] - leftmost_pt[0]
            # Note: Image coordinates: Positive dy means downwards on the screen
            dy = rightmost_pt[1] - leftmost_pt[1]

            if abs(dx) > MIN_DX_FOR_SLOPE: # Check if segment is wide enough horizontally
                slope = dy / dx
                line_angle_deg = math.degrees(math.atan2(-dy, dx)) # Angle CCW from positive X axis

                # If the line goes up on screen (left to right), dy is negative, slope is negative
                if slope < -VERTICAL_SLOPE_THRESHOLD: # Significantly upwards slope on screen
                    command_v = "UP"
                # If the line goes down on screen (left to right), dy is positive, slope is positive
                elif slope > VERTICAL_SLOPE_THRESHOLD: # Significantly downwards slope on screen
                    command_v = "DOWN"
                else: # Mostly horizontal slope
                    command_v = "HOVER_V"
            else:
                # Line is near vertical or very short horizontally
                # Use centroid's vertical position relative to center as fallback
                if cy < frame_center_y - 30: # Above center (adjust threshold as needed)
                     command_v = "UP"
                elif cy > frame_center_y + 30: # Below center (adjust threshold as needed)
                     command_v = "DOWN"
                else:
                     command_v = "HOVER_V" # Vertically centered

            # --- Combine Commands (Prioritize Turning) ---
            if command_h != "HOVER_H":
                final_command = command_h # Turn if needed
            elif command_v != "HOVER_V":
                final_command = command_v # Move vertically if horizontally ok
            else:
                final_command = "HOVER" # Stay put if centered and flat segment

        else:
             # Centroid calculation failed (m00 was zero)
             line_found = False
             final_command = "HOVER" # Or maybe SEARCH? Hover seems safer.

    else:
        # No significant contour found
        line_found = False
        final_command = "SEARCH"

    # --- Send Command to Console ---
    send_drone_command(final_command)

    # --- Display Information ---
    # Draw target X and deadzone
    cv2.line(frame, (target_cx, 0), (target_cx, frame_height), (0, 255, 0), 1) # Green target X
    left_bound_viz = target_cx - int(HORIZONTAL_DEADZONE_FRACTION * frame_width / 2)
    right_bound_viz = target_cx + int(HORIZONTAL_DEADZONE_FRACTION * frame_width / 2)
    cv2.line(frame, (left_bound_viz, 0), (left_bound_viz, frame_height), (255, 255, 0), 1) # Cyan left bound
    cv2.line(frame, (right_bound_viz, 0), (right_bound_viz, frame_height), (255, 255, 0), 1) # Cyan right bound


    cv2.putText(frame, f"Command: {final_command}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if line_found:
        cv2.putText(frame, f"Centroid: ({cx}, {cy}) TargetX: {target_cx}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(frame, f"L:{leftmost_pt} R:{rightmost_pt}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(frame, f"Angle: {line_angle_deg:.1f} deg", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(frame, f"Threshold: {GRAYSCALE_THRESHOLD}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    else:
         cv2.putText(frame, "Line not found", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
         cv2.putText(frame, f"Threshold: {GRAYSCALE_THRESHOLD}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)


    cv2.imshow("Black Line Mask (Refined)", refined_mask)
    cv2.imshow("Line Following View (Black Line)", frame)

    # --- Exit Condition (Press ESC) ---
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        print("Escape hit, closing...")
        break

# --- Cleanup ---
send_drone_command("HOVER") # Ensure hover command is sent before exiting
tello.streamoff()   
cv2.destroyAllWindows()