from tello_wrapper import TelloWrapper as Tello
import cv2
import logging
import time
import os # Import os for setting env variable

# Force XCB platform (even if you think it's not Qt, let's rule it out)
os.environ['QT_QPA_PLATFORM'] = 'xcb'

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

me = Tello()
me.connect()
print(me.get_battery())
me.stream()