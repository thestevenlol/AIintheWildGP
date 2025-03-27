import cv2
import numpy as np

# --- Constants (Adjust these!) ---

# HSV Blue Range - Lowered S and V minimums slightly
# TRY ADJUSTING THESE RANGES BASED ON YOUR ACTUAL BLUE MARKER AND LIGHTING
# Hue: 100-130 is typical, but adjust if needed.
# Saturation: Lowered min from 50 to 40 (allows slightly less saturated blue)
# Value: Lowered min from 50 to 40 (allows slightly darker blue)
LOWER_GREEN = np.array([35, 40, 40]) # ADJUST HUE (100) and MIN S/V (40, 40)
UPPER_GREEN = np.array([85, 255, 255])# ADJUST HUE (130)

# Morphological Kernel Size - Reduced for finer detail
KERNEL_SIZE = 2 # Reduced from 5 to 3

# Morphological Iterations - Reduced for less aggressive closing
MORPH_ITERATIONS = 1 # Reduced from 2 to 1

# --- Webcam Setup ---
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Error: Could not open webcam.")
    exit()

cv2.namedWindow("Blue Mask (Refined)")
# Optional: Window to see the mask *before* refinement
# cv2.namedWindow("Blue Mask (Initial)")

# --- Processing Loop ---
while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame")
        break

    # --- Optional Pre-processing: Gaussian Blur ---
    # Applying a small blur can sometimes help reduce noise before thresholding
    # Uncomment the next line to try it. Adjust kernel size (e.g., (3,3)) if needed.
    # blurred_frame = cv2.GaussianBlur(frame, (3, 3), 0)
    # hsv_image = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
    # --- End Optional Blur ---

    # If not using blur, convert the original frame
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # Use this line if blur is commented out

    # 3. Create Green Mask
    initial_mask = cv2.inRange(hsv_image, LOWER_GREEN, UPPER_GREEN)

    # Optional: Display initial mask to see effect of threshold change
    # cv2.imshow("Blue Mask (Initial)", initial_mask)

    # 4. Refine Mask (Morphological Closing - Less Aggressive)
    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
    refined_mask = cv2.morphologyEx(initial_mask, cv2.MORPH_CLOSE, kernel, iterations=MORPH_ITERATIONS)

    # --- Optional Alternative/Additional Refinement ---
    # If closing isn't enough or removes too much, you could try:
    # 1. Opening first to remove noise:
    #    open_kernel = np.ones((3,3), np.uint8) # Small kernel for noise removal
    #    opened_mask = cv2.morphologyEx(initial_mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    #    refined_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel, iterations=MORPH_ITERATIONS) # Then close
    # 2. Dilation after closing to thicken lines slightly if they become too thin:
    #    refined_mask = cv2.dilate(refined_mask, kernel, iterations=1)
    # --- End Optional Refinement ---

    # 5. Apply Mask to Original Frame
    result_image = cv2.bitwise_and(frame, frame, mask=refined_mask)

    cv2.imshow("Blue Mask (Refined)", refined_mask) # Renamed window

    # 7. Exit Condition (Press ESC)
    k = cv2.waitKey(1)
    if k % 256 == 27:
        print("Escape hit, closing...")
        break

# --- Cleanup ---
cam.release()
cv2.destroyAllWindows()