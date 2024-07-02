import cv2
import time
import numpy as np
import math
import depthai as dai
import PySimpleGUI as sg

from dominh.src.dominh import connect

def read_set_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            # Read the content of the file
            content = file.read().strip()

            # Remove any leading/trailing whitespace and newline characters
            content = content.strip()

            # Assuming the content of the file is a string representation of a set
            # Convert the string back to a set object using eval
            # Note: Be cautious when using eval with user input as it can be a security risk
            return eval(content)
    except FileNotFoundError:
        print("File not found.")
        return None
    except Exception as e:
        print("An error occurred:", e)
        return None
def gen_gui_window():
    # Define the layout of the popup window
    layout = [
        [sg.Text('Enter filepath:')],
        [sg.InputText(key='-FILEPATH-', size=(35, 1)), sg.FileBrowse()],
        [sg.Button('Go'), sg.Button('Cancel'), sg.Button('Calibrate'), sg.Button('Trigger'), sg.Button('Load')]
    ]

    # Create the popup window
    return sg.Window('Pennies', layout, finalize=True)
def wait_for_trigger(c):
    while True:
        ack = c.numreg(1).val
        if ack == 1:
            print('Trigger recieved. Starting process...')
            c.numreg(1).val = -1
            return True
def connect_to_robot(ip):
    try:
        c = connect(ip, karel_auth=('user', 'pass'))
        print('Connected to robot host.')
        return c
    
    except Exception as e:
        print(f'Error: {e}')
        return None
def average_tuple(tuple_list):
    if not tuple_list:
        return None  # If the list is empty, return None
    dimensions = len(tuple_list[0])  # Number of dimensions in the tuples
    sum_tuple = [0] * dimensions  # Initialize a list to hold the sum of each dimension
    for tup in tuple_list:
        sum_tuple = [sum(x) for x in zip(sum_tuple, tup)]  # Add each dimension of the tuple to the sum
    avg_tuple = tuple(val / len(tuple_list) for val in sum_tuple)  # Calculate the average for each dimension
    print('\nAVG SHEEN:',avg_tuple)
    return avg_tuple
def get_matches(checkpoint, bitmap):
    if checkpoint is not None:
        matches = checkpoint
        # white_pixel_locations = find_white_pixels(bitmap)
        # if white_pixel_locations != []:
        #     for location in white_pixel_locations:
        #         if not (location in matches):
        #             matches.add(location)
    else:
        matches = set()
        # white_pixel_locations = find_white_pixels(bitmap)
        # if white_pixel_locations != []:
        #     for location in white_pixel_locations:
        #         matches.add(location)  # Using set for faster lookup
    return matches
def find_matching_locations(bitmap, penny_val, tolerance):
    sums = []
    h,w,c_ = bitmap.shape
    ite = 0
    for x in range(w):
        for y in range(h):
            map_val = bitmap[y, x]
            color_dist = np.linalg.norm(map_val - penny_val)
            if color_dist < tolerance:
                sums.append([color_dist,(y,x)])
            ite = ite+1
    sums.sort(key=lambda x: x[0])

    return sums
def find_white_pixels(img):
    white_pixels = []
    height, width, ch = img.shape
    for y in range(height):
        for x in range(width):
           pixel = img[y,x]
           if all(value == 255 for value in pixel):
                white_pixels.append((y, x))
    return white_pixels
def repick(c):

    c.numreg(4).val = -1
    c.numreg(5).val = -1
    c.numreg(6).val = -1
    c.numreg(7).val = -1
    c.numreg(8).val = -1
    c.numreg(2).val = 1
    print('REPICK')
def wait_for_trigger(c):

    while c.numreg(1).val != 1:
        pass
    print('Trigger recieved. Starting process...')
    c.numreg(1).val = -1
    return True 
def find_pennies(average_image, pennies_in_frame, PX_AVG_RAD, width, height, LOW, HIGH):

    gray = average_image.copy()
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    # gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    # # Apply GaussianBlur to reduce noise
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Use HoughCircles to detect circles
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=130,
                            param1=90, param2=40, minRadius=65, maxRadius=75)
    # Ensure at least some circles were found
    if circles is not None:
        # Convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # Loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:

            if x > 0:
                x -= 1
            if y > 0:
                y -= 1

            if (y-PX_AVG_RAD >= 0) and (y+PX_AVG_RAD < height) and (x-PX_AVG_RAD >= 0) and (x+PX_AVG_RAD < width):
                roi = average_image[y-PX_AVG_RAD:y+PX_AVG_RAD, x-PX_AVG_RAD:x+PX_AVG_RAD]
                # Calculate the average pixel intensity in the ROI
                # sheen = np.mean(roi)
                sheen = np.mean(roi, axis=(0, 1))

                # if sheen < LOW:
                #     sheen = 0
                # elif sheen > HIGH:
                #     sheen = 255
                # else:
                #     DIFF = HIGH-LOW
                #     sheen = int( ((sheen-LOW)/DIFF)*255 )

                pennies_in_frame.append(Coin(sheen,x,y,r))
            else:
                sheen = average_image[y,x]

                # if sheen < LOW:
                #     sheen = 0
                # elif sheen > HIGH:
                #     sheen = 255
                # else:
                #     DIFF = HIGH-LOW
                #     sheen = int( ((sheen-LOW)/DIFF)*255 )

                pennies_in_frame.append(Coin(sheen,x,y,r))

                #     pennies_in_frame.append(Coin(sheen,x,y,r))
                # else:
                #     sheen = average_image[y,x]
                #     pennies_in_frame.append(Coin(sheen,x,y,r))   
                
    return pennies_in_frame
def find_extreme_pixel_values(image_list):

    # Initialize variables to keep track of the extreme pixel values
    largest_intensity = 0
    smallest_intensity = float('inf')  # Initialize with a very large number

    # Iterate through each pixel in the image
    for pixel in image_list:
        # Extract B, G, R values from the pixel
        intensity = pixel

        # Update the largest and smallest sum if necessary
        if intensity > largest_intensity:
            largest_intensity = intensity
        if intensity < smallest_intensity:
            smallest_intensity = intensity

    return smallest_intensity, largest_intensity
def average_images(image_list):
    # Convert image_list to a NumPy array
    images_array = np.array(image_list)
    # Calculate the average of all the grayscale images along the first axis (axis=0)
    average_gray_image = np.mean(images_array, axis=0).astype(np.uint8)
    
    return average_gray_image
def scan(qRgb):

    image_list = []
    t0 = time.time()
    while time.time()-t0 < 0.5:
        inRgb = qRgb.get()
        frame = inRgb.getCvFrame()
        image_list.append(frame)

    return average_images(image_list)
def start(bitmap, LOW, HIGH, c, qRgb, cam1, device, checkpoint):

    last_checkpoint = None

    exp, iso = 900, 150

    ite = 0
    repick_count = 0
    num_repicks = 0

    TOLERANCE = 40

    # tolerance_hsv = np.array([50, 200, 200])

    X_BOUND = 2500
    Y_BOUND = 900

    # In millimeters
    DIAMETER_OF_PENNY = 19.05
    RADIUS_OF_PENNY = DIAMETER_OF_PENNY/2
    X_ORIGIN = -127.1#-123.809
    Y_ORIGIN = 315.2#315.769
    # PX conversion
    PX_PER_MM = 7.559
    # Image size
    height, width, ch = bitmap.shape
    # Sheen
    PX_AVG_RAD = math.floor( (RADIUS_OF_PENNY*PX_PER_MM) / np.sqrt(2) )

    matches = get_matches(checkpoint, bitmap)

    pennies_in_frame = []
    average_image = None

    print('TOTAL PENNIES TO PLACE:', (width*height)-len(matches))

    # BEGIN PROCESS
    EXIT = False
    START_TIMER = True

    while not EXIT:

        if START_TIMER:
            time1 = time.time()

        DATA_SENT = False
        REPICK = False

        wait_for_trigger(c)

        PLACE_X = 0
        PLACE_Y = 0
        PICK_X = 0
        PICK_Y = 0

        average_image = scan(qRgb)
        h,w,c_ = average_image.shape
        pennies_in_frame = []
        pennies_in_frame = find_pennies(average_image, pennies_in_frame, PX_AVG_RAD, w, h, LOW, HIGH)
        print('PENNIES:', len(pennies_in_frame))

        for p in range(len(pennies_in_frame)):

            penny_hue = pennies_in_frame[p].sheen
            
            matching_locations = find_matching_locations(bitmap, penny_hue, TOLERANCE)
            
            if matching_locations is not None:
                if len(matching_locations) > 0:
                    for match_location in matching_locations:
                        match_location = tuple(match_location[0:2])
                        if match_location[1] not in matches:
                            matches.add(match_location[1])
                            PLACE_Y, PLACE_X = match_location[1]
                            PLACE_X = float((PLACE_X*(DIAMETER_OF_PENNY+3))+RADIUS_OF_PENNY)
                            PLACE_Y = float((PLACE_Y*(DIAMETER_OF_PENNY+3))+RADIUS_OF_PENNY)
                            break

            if PLACE_X != 0 and PLACE_Y != 0 and PLACE_X < X_BOUND and PLACE_Y < Y_BOUND:
                
                lens_distortion = float((pennies_in_frame[p].x / w) * 5)
                PICK_X = X_ORIGIN + float((pennies_in_frame[p].x)/PX_PER_MM) + 2.75 + lens_distortion
                PICK_Y = Y_ORIGIN + float((pennies_in_frame[p].y)/PX_PER_MM) + 2.75

                num_of_pennies = len(pennies_in_frame)

                cv2.circle(average_image, (pennies_in_frame[p].x, pennies_in_frame[p].y), pennies_in_frame[p].r, (0,0,255), 3)
                cv2.rectangle(average_image,
                              (pennies_in_frame[p].x-PX_AVG_RAD, pennies_in_frame[p].y-PX_AVG_RAD),
                              (pennies_in_frame[p].x+PX_AVG_RAD, pennies_in_frame[p].y+PX_AVG_RAD),
                              (0,0,255), 2)

                if PICK_X != 0:
                    c.numreg(4).val = PICK_X
                if PICK_Y != 0:
                    c.numreg(5).val = PICK_Y
                c.numreg(6).val = PLACE_X
                c.numreg(7).val = PLACE_Y
                c.numreg(8).val = num_of_pennies

                c.numreg(2).val = 1

                print('\nDATA:', c.numreg(4).val,c.numreg(5).val,c.numreg(6).val,c.numreg(7).val,num_of_pennies)
                print('SHEEN VALUES:', 'Penny -', pennies_in_frame[p].sheen, '||| Bitmap -', bitmap[match_location[1]])
                print('BITMAP Location:', match_location[1])

                last_checkpoint = matches
                with open(r'\last_checkpoint\checkpoint.txt', 'w') as file:
                    file.write(str(last_checkpoint))

                time2 = time.time()
                print('\nEstimated finish time:', str(round( (((time2-time1)*( (width*height)-len(matches) )) / 3600), 2)), 'hours' )
                START_TIMER = True

                DATA_SENT = True
                break

        if not DATA_SENT:
            repick(c)
            repick_count += 1
            num_repicks += 1
            REPICK = True
        else:
            num_repicks = 0

        if REPICK:
            START_TIMER = False
            if (num_repicks >= 10) and (num_repicks < 20):
                TOLERANCE += 20
                print('NEW HUE TOLERANCE:', TOLERANCE)
            if num_repicks > 20:
                EXIT = True

        ite += 1
        print('\nITE:',ite)
        print('ReImage count:', repick_count)
        print('Num of pennies left:', (width*height)-len(matches))
        print('consecutive repicks:', num_repicks)

    print('Final tolerance', TOLERANCE)
    return

class Camera:

    def __init__(self, ip):

        self.pipeline = dai.Pipeline()
        self.cam_ip = ip
        self.device_info = dai.DeviceInfo(self.cam_ip)

        # Create pipelines
        camRgb = self.pipeline.create(dai.node.ColorCamera)

        # Properties
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        camRgb.setInterleaved(False)
        camRgb.setPreviewSize(1920, 1080)

        #create focus control
        self.controlIn = self.pipeline.createXLinkIn()
        self.controlIn.setStreamName('control1')
        self.controlIn.out.link(camRgb.inputControl)

    def set_focus(self, device, desired_focus):
        # Set autofocus / exposure time
        controlQueue1 = device.getInputQueue('control1')
        ctrl1 = dai.CameraControl()
        ctrl1.setAutoFocusMode(dai.CameraControl.AutoFocusMode.OFF)
        ctrl1.setManualFocus(desired_focus)
        
        controlQueue1.send(ctrl1)
        print("\nAuto Focus OFF")

    def initialize_queue(self, device):
        return device.getOutputQueue(name="rgb1", maxSize=1, blocking=False)

    def set_exposure(self, device, exp, iso):
        # Set autofocus / exposure time
        controlQueue1 = device.getInputQueue('control1')
        ctrl1 = dai.CameraControl()
        ctrl1.setManualExposure(exp,iso)
        controlQueue1.send(ctrl1)

    def scan(self,qRgb):
        image_list = []

        t0 = time.time()
        while time.time()-t0 < 2:
            inRgb = qRgb.get()
            frame = inRgb.getCvFrame()
            image_list.append(frame)

        return average_images(image_list)

    def calibrate(self, PX_AVG_RAD, qRgb):
        LOW,HIGH = 0,0
        print('calibrating...')
        pennies_in_frame = []

        average_image = self.scan(qRgb)
        h,w,c = average_image.shape
        pennies_in_frame = find_pennies(average_image, pennies_in_frame, PX_AVG_RAD, w, h)

        sheen_list = []
        for p_ in pennies_in_frame:
            sheen_list.append(p_.sheen)
        if len(sheen_list) > 0:
            LOW,HIGH = find_extreme_pixel_values(sheen_list)
        else:
            LOW,HIGH = 0,255

        print('NORMALIZAED PENNY SHEEN MIN-MAX:', LOW,'|',HIGH)
        return LOW, HIGH
    def trigger(self, qRgb):
        inRgb = qRgb.get()
        frame = inRgb.getCvFrame()
        hh,ww,cc = frame.shape
        trig_img = r'C:\Users\bkraw\Desktop\PennyCollage\images_bitmap\frame.jpg'
        cv2.imwrite(trig_img, frame)
        print(f'Frame captured. Saved to {trig_img}')

        self.pennies = []
        PX_PER_MM = 7.559
        DIAMETER_OF_PENNY = 19.05
        RADIUS_OF_PENNY = DIAMETER_OF_PENNY/2
        PX_AVG_RAD = math.floor( (RADIUS_OF_PENNY*PX_PER_MM) / np.sqrt(2) )
        self.pennies = find_pennies(frame, self.pennies, PX_AVG_RAD, ww, hh)
        # sheens = []
        # for p in range(len(self.pennies)):
        #     sheens.append(self.pennies[p].sheen)
        
        # avg_sheen = average_tuple(sheens)
class Coin:

    def __init__(self, sheen, x, y, r):
        self.diameter = r*2
        self.sheen = []
        self.x = x
        self.y = y
        self.r = r

        for s in sheen:
            self.sheen.append(int(s))

        self.sheen = tuple(self.sheen)

        # print('SHEEN:', self.sheen)

    # def __init__(self, sheen, x, y, r):
    #     self.diameter = r*2
    #     self.sheen = sheen
    #     self.x = x
    #     self.y = y
    #     self.r = r


LOW,HIGH = 93,200
cam1 = Camera('10.0.0.101')

END = False

checkpoint = None

window = gen_gui_window()

# Initialize first camera pipeline
with dai.Device(cam1.pipeline, cam1.device_info) as device1:

    focus = 145
    cam1.set_focus(device1, focus)
    cam1.set_exposure(device1,900,150)
    qRgb = cam1.initialize_queue(device1)

    while not END:
        event, values = window.read()

        ### EVENT ###
        if event == sg.WINDOW_CLOSED or event == 'Cancel':
            print('Thank you')
            END = True
            break

        ### EVENT ###
        elif event == 'Go':
            filepath = values['-FILEPATH-']
            
            # Validate the input values
            if filepath:

                filepath = str(filepath).replace('"', '')
                filepath = filepath.replace('\'', '')
                
                # Read the bitmap image
                bitmap = cv2.imread(str(filepath))
                height, width, ch = bitmap.shape

                break
                
            else:
                sg.popup_error('Please enter valid filepath!')

        ### EVENT ###
        elif event == 'Calibrate':

            PX_PER_MM = 7.559
            DIAMETER_OF_PENNY = 19.05
            RADIUS_OF_PENNY = DIAMETER_OF_PENNY/2
            PX_AVG_RAD = math.floor( (RADIUS_OF_PENNY*PX_PER_MM) / np.sqrt(2) )

            LOW,HIGH = cam1.calibrate(PX_AVG_RAD, qRgb)

        ### EVENT ###
        elif event == 'Trigger':

            cam1.trigger(qRgb)

        ### EVENT ###
        elif event == 'Load':

            
            load_filepath = r'\last_checkpoint\checkpoint.txt'
            checkpoint = read_set_from_file(load_filepath)

    window.close()

    if not END:
        print('Camera Initialized')

        robot_ip = '10.0.0.10'
        c = connect_to_robot(robot_ip)
        if c is not None:
            start(bitmap, LOW, HIGH, c, qRgb, cam1, device1, checkpoint)