from periodics import PeriodicSleeper
import serial.tools.list_ports
import pygame
import os
import time
import csv
from datetime import datetime
import re
import base64, struct


PORT = "/dev/cu.usbmodem1101" # for mac

ser = None      #serial port for estop
done = False    #exit flag, set to true if pygame window is closed    


# JOYSTICK
joy_data = {
    'leftx': 0,
    'lefty': 0,
    'rightx': 0,
    'righty': 0,
    'lefttrigger': 0,
    'righttrigger': 0,
    'A': 0,
    'B': 0,
    'X': 0,
    'Y': 0,
    '-': 0,
    'home': 0,
    '+': 0,
    'leftstickbutton': 0,
    'rightstickbutton': 0,
    'leftbumper': 0,
    'rightbumper': 0,
    'dpadup': 0,
    'dpaddown': 0,
    'dpadleft': 0,
    'dpadright': 0,
    'circle': 0,
}
axes_calibrated_dict = {
    'leftx+': False,
    'leftx-': False,
    'lefty+': False,
    'lefty-': False,
    'rightx+': False,
    'rightx-': False,
    'righty+': False,
    'righty-': False
}
axes_calibrated = False

# Runs every main loop
def handle_joysticks():
    global axes_calibrated
    global axes_calibrated_dict

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            global done
            done = True

        joysticks = {}
        if event.type == pygame.JOYDEVICEADDED:
            joy = pygame.joystick.Joystick(event.device_index)
            joysticks[joy.get_instance_id()] = joy
            print(f"Joystick {joy.get_instance_id()} connected")
            axes_calibrated = False

        if event.type == pygame.JOYDEVICEREMOVED:
            del joysticks[event.instance_id]
            print(f"Joystick {event.instance_id} disconnected")

        for joystick in joysticks.values():
            joy_data['leftx'] = joystick.get_axis(0) 
            joy_data['lefty'] = -joystick.get_axis(1)
            joy_data['rightx'] = joystick.get_axis(2)
            joy_data['righty'] = -joystick.get_axis(3) # -1 to 1
            joy_data['lefttrigger'] = joystick.get_axis(4) # -1 to 1
            joy_data['righttrigger'] = joystick.get_axis(5) # -1 to 1
            joy_data['A'] = joystick.get_button(0)
            joy_data['B'] = joystick.get_button(1)
            joy_data['X'] = joystick.get_button(2)
            joy_data['Y'] = joystick.get_button(3)
            joy_data['-'] = joystick.get_button(4)
            joy_data['home'] = joystick.get_button(5) #make sure to disable Launchpad https://apple.stackexchange.com/questions/458669/how-do-i-disable-the-home-button-on-a-game-controller-in-macos
            joy_data['+'] = joystick.get_button(6)
            joy_data['leftstickbutton'] = joystick.get_button(7)
            joy_data['rightstickbutton'] = joystick.get_button(8)
            joy_data['leftbumper'] = joystick.get_button(9)
            joy_data['rightbumper'] = joystick.get_button(10)
            joy_data['dpadup'] = joystick.get_button(11)
            joy_data['dpaddown'] = joystick.get_button(12)
            joy_data['dpadleft'] = joystick.get_button(13)
            joy_data['dpadright'] = joystick.get_button(14)
            joy_data['circle'] = joystick.get_button(15)

            for axis_name in ['leftx', 'lefty', 'rightx', 'righty']: #deadband
                if(abs(joy_data[axis_name]) < 0.02):
                    joy_data[axis_name] = 0


            #seems like pygame needs to know the max axis value so need to move stick to max/min
            for axis_name in ['leftx', 'lefty', 'rightx', 'righty']:
                if(joy_data[axis_name] > 0.99):
                    axes_calibrated_dict[f"{axis_name}+"] = True
                elif(joy_data[axis_name] < -0.99):
                    axes_calibrated_dict[f"{axis_name}-"] = True
            
            #check if each axis calibrated, complete only all done and at home
            if(not axes_calibrated):
                axes_calibrated_try = True
                for calibration_label in axes_calibrated_dict:
                    if(axes_calibrated_dict[calibration_label] == False):
                        axes_calibrated_try = False
                        break
                for axis_name in ['leftx', 'lefty', 'rightx', 'righty']:
                    if(abs(joy_data[axis_name]) > 0.01):
                        axes_calibrated_try = False
                        break
                axes_calibrated = axes_calibrated_try
                if(axes_calibrated):
                    joystick.rumble(0.7, 0.0, 100) #indicate calibration done
                else:
                    for axis_name in ['leftx', 'lefty', 'rightx', 'righty']:
                        joy_data[axis_name] = 0


            break #assume only one joystick

    if(not axes_calibrated):
        print("please calibrate joysticks")
        print("move axes in a circle")


def main():
    global done, ser, joy_data

    # Start pygame in order to read joysticks
    os.environ['SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS'] = '1' #makes sure that joystick still reads if user is not focused on pygame window
    pygame.init()
    pygame.display.set_mode((50, 50))
    

    # Data received from rock is interpreted using regex and stored in the telemetry dictionary
    telemetry = {}
    rock_telemetry_labels = [
        'battery_filt',
        'mot_angle',
        'mot_angvel',
        'thetaM_motor',
        'quat_imu[0]',
        'quat_imu[1]',
        'quat_imu[2]',
        'quat_imu[3]',
        'angvel_imu[0]',
        'angvel_imu[1]',
        'angvel_imu[2]',
    ]
    pattern = re.compile( # regex for parsing telemetry string returned from estop
        r'(?m)^[A-Za-z_]+:(\d+)\r?\n[A-Za-z_]+:(\d+)\r?\n[A-Za-z_]+:(\d+)\r?\n(.+)$' # 3 lines of label:int\n followed by a string
    )


    # Prepare to write telemetry and joystick data into csv file
    os.makedirs('cmdlogs', exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M')
    filename = f'cmdlogs/{timestamp}.csv'
    csvfile = open(filename, 'w', newline='')
    fieldnames = rock_telemetry_labels + list(joy_data.keys())
    csvwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
    csvwriter.writeheader()


    # Start another thread to send cmd_str to estop periodically
    cmd_str = ""    #string to send to estop, will be overwritten later in main loop
    def send_to_estop():
        global ser, cmd_str
        if(ser is None):
            return
        try:
            ser.write(cmd_str.encode())
        except Exception as e:
            print(e)
            ser = None
            print("Estop disconnected")
    PeriodicSleeper(send_to_estop, 0.01)
        

    # Main loop
    last_time = time.time()
    while not done:
        time.sleep(0.01) # makes loop run a bit slower than 100Hz
        dt = time.time() - last_time
        last_time = time.time()
        print(f"dt: {dt:.5f}") 


        # Read joystick data and puts it into joy_data
        handle_joysticks()
        

        # Establish serial connection with estop, wait until established
        while(ser is None):
            try:
                ser = serial.Serial(PORT, baudrate=115200, timeout=1, stopbits=serial.STOPBITS_TWO)
            except:
                ser = None
                print("plug in estop pls")
                time.sleep(0.5)
    

        # Reads from estop serial port to get connection status and update telemetry.
        # Expected string received from estop: "station_ok:1\nespnow_ok:1\nestopped:0\n<base64 encoded float array>\n"
        s = ser.read_all().decode('utf-8', errors='ignore')
        m = pattern.search(s)
        if m:
            telemetry['station_ok'] = m.group(1)
            telemetry['espnow_ok'] = m.group(2)
            telemetry['estopped'] = m.group(3)
            payload = m.group(4)
            try:
                raw = base64.b64decode(payload)
                n = len(raw) // 4
                floats = struct.unpack(f'<{n}f', raw)
                for i in range(len(floats)):
                    telemetry[rock_telemetry_labels[i]] = floats[i]
            except Exception as e:
                print(e)
        

        # Print telemetry received from estop
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~\nTelemetry")
        for key in telemetry:
            if isinstance(telemetry[key], float):  # Print floats with 3 decimal places
                print(f"{key}: {telemetry[key]:0.3f}")
            else:
                print(f"{key}: {telemetry[key]}")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


        # Generate commands using joystick data
        cmdx = int(joy_data['rightx'] * 4096 + 4096)
        cmdy = int(joy_data['righty'] * 4096 + 4096)
        leftx = int(joy_data['leftx'] * 4096 + 4096)
        lefty = int(joy_data['lefty'] * 4096 + 4096)
        run0 = 0
        if joy_data['rightbumper']:
            run0 += 1
        if joy_data['righttrigger']>0.5:
            run0 += 2


        # Builds cmd_str to send to estop serial port, actually sent by send_to_estop function
        # Expected string: "cmdx:4096\ncmdy:4096\nleftx:4096\nlefty:4096\nrun0:0\n"
        try:
            if(telemetry['estopped'] == 1): run0 = 0
        except Exception as e:
            print('nodata', e)
        cmd_str = ""
        cmd_str += f"cmdx:{cmdx:05}\n"
        cmd_str += f"cmdy:{cmdy:05}\n"
        cmd_str += f"leftx:{leftx:05}\n"
        cmd_str += f"lefty:{lefty:05}\n"
        cmd_str += f"run0:{run0:05}\n"
        cmd_str += "#\t"
        print(cmd_str)


        # Write telemetry and joystick data into csv file
        combined_dict = {**telemetry, **joy_data}
        csvwriter.writerows([combined_dict])


    # Close csv file once main loop is done
    csvfile.close()


if __name__ == "__main__":
    main()