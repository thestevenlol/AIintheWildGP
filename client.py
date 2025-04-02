import socket
import time
import sys
import threading
import cv2 as 
import subprocess
import os
from time import sleep
from djitellopy import Tello
from tello_stream import TelloVideoStream
from cvfpscalc import CvFpsCalc

if __name__ == "__main__":
    tello = Tello()
    tello.connect()
    tello.streamon()
    
    cap = tello.get_frame_read()
    cv_fps_calc = CvFpsCalc(buffer_len=10)

    while True:
        fps = cv_fps_calc.get()

        # Process Key (ESC: end)
        key = cv.waitKey(1) & 0xff
        if key == 27:  # ESC
            break
        elif key == 32:  # Space
            if not in_flight:
                # Take-off drone
                tello.takeoff()
                in_flight = True

            elif in_flight:
                # Land tello
                tello.land()
                in_flight = False

        elif key == ord('k'):
            mode = 0
            KEYBOARD_CONTROL = True
            WRITE_CONTROL = False
            tello.send_rc_control(0, 0, 0, 0)  # Stop moving
        elif key == ord('g'):
            KEYBOARD_CONTROL = False
        elif key == ord('n'):
            mode = 1
            WRITE_CONTROL = True
            KEYBOARD_CONTROL = True

        if WRITE_CONTROL:
            number = -1
            if 48 <= key <= 57:  # 0 ~ 9
                number = key - 48

        # Camera capture
        image = cap.frame

        # Battery status and image rendering
        cv.putText(debug_image, "Battery: {}".format(battery_status), (5, 720 - 5),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv.imshow('Tello Gesture Recognition', debug_image)

    sleep(2)
    tello.streamoff()
    tello.end()
    print("Tello stream off")