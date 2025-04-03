from djitellopy import Tello

tello = Tello()
tello.connect()
tello.takeoff()

while True: 
    user_input = input("Enter xyz coordinates (x y z) or 'exit' to quit: ")
    if user_input.lower().strip() == 'exit' or user_input.lower().strip() == 'land':
        tello.land()
        break
    try:
        x, y, z = map(int, user_input.split())
        tello.go_xyz_speed(x, y, z, 10)
    except ValueError:
        print("Invalid input. Please enter three integers separated by spaces.")
    except Exception as e:
        print(f"An error occurred: {e}")
    
