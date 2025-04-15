from periodics import PeriodicSleeper
import serial.tools.list_ports
import pygame
import os
import time

os.environ['SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS'] = '1'
pygame.init()
screen = pygame.display.set_mode((50, 50))
port = "/dev/cu.usbmodem1101" # for mac
ser = None
done = False

cmd_str = ""
message = '' # received
messagebuffer = ''
telemetry = {
    'v': 0
}
delimiter = '\t'

wheel_speed = 0

def main():
    global done, ser, cmd_str

    send_handler = PeriodicSleeper(send_to_estop, 0.01)
    while not done:
        while(ser is None):
            try:
                ser = serial.Serial(port, baudrate=115200, timeout=1, stopbits=serial.STOPBITS_TWO)
            except:
                ser = None
                print("plug in estop pls")
                time.sleep(0.5)
        
        handle_joysticks()
        recv_from_estop() #updates joy_data

        cmdx = int(joy_data['rightx'] * 4096 + 4096)
        cmdy = int(joy_data['righty'] * 4096 + 4096)
        leftx = int(joy_data['leftx'] * 4096 + 4096)
        lefty = int(joy_data['lefty'] * 4096 + 4096)

        run0 = 0
        if joy_data['rightbumper']:
            run0 += 1
        if joy_data['righttrigger']>0.5:
            run0 += 2

        

        # if "estopped:0" in message:
        #     run = 0
        try:
            if(telemetry['estopped'] == 1): run = 0
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

        time.sleep(0.01)




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


def recv_from_estop():
    # print(ser.read_all().decode("utf-8", errors='ignore'), end=None)
    global ser
    global messagebuffer
    global message
    global axes_calibrated

    messagecount = 0

    try:
        if ser.in_waiting > 0:
            uarttext = ser.read_all().decode('utf-8', errors='ignore')
            
            ending=0
            while(uarttext):
                ending = uarttext.find(delimiter)
                if(ending == -1):
                    break

                message += uarttext[0:ending]

                print(message)            # Uncomment to print message to terminal

                messagebuffer = message
                # print(messagebuffer)

                lines = messagebuffer.split('\n')
                for line in lines:
                    parts = line.split(':')
                    key = parts[0].strip()
                    try:
                        value = float(parts[1].strip())
                    except Exception:
                        continue
                    telemetry[key] = value

                messagecount += 1
                message = "" #clear message
                uarttext = uarttext[ending+len(delimiter):] #front of buffer used up

            message = uarttext #whatver is left over
        
    except Exception as e:
        print(e)
        return
    

# JOYSTICK

joysticks = {}
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

global axes_calibrated
axes_calibrated = False

def handle_joysticks():
    global axes_calibrated
    global axes_calibrated_dict

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            global done
            done = True

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


if __name__ == "__main__":
    main()