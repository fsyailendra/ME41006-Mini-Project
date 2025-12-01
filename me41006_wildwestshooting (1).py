import cv2
import numpy as np
import Adafruit_PCA9685
import time
import RPi.GPIO as GPIO

# Initialize servo motor
pwm = Adafruit_PCA9685.PCA9685(0x41)
pwm.set_pwm_freq(50)

def set_servo_angle(channel, angle):
    pulse = 4096 * ((angle * 11) + 500) / 20000
    pwm.set_pwm(channel, 0, int(pulse))

# GPIO setup for shooting mechanism
IN1 = 24
IN2 = 23
ENA = 18

# Set up GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(ENA, GPIO.OUT)

def motor_on():
    """Turn on the shooting motor"""
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(ENA, GPIO.HIGH)
    print('Motor ON - SHOOTING!')

def motor_off():
    """Turn off the shooting motor"""
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(ENA, GPIO.LOW)
    print('Motor OFF')

def shoot():
    """Activate shooting mechanism for 0.3 seconds"""
    motor_on()
    time.sleep(0.3)
    motor_off()

# Reset to starting servo position
set_servo_angle(1, 90)
set_servo_angle(2, 86)
set_servo_angle(3, 90)

# Initialize shooting mechanism to OFF
motor_off()

# Camera setup - lower resolution = faster processing
cap = cv2.VideoCapture(0)
# Reduce resolution for speed
cap.set(3, 320)  # Reduced from 640 to 320
cap.set(4, 240)  # Reduced from 480 to 240
cap.set(cv2.CAP_PROP_FPS, 60)  # Try for 60 FPS if the camera supports it
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency

# HSV range for the green LED target - tuned over many attempts
green_lower = np.array([35, 100, 100])
green_upper = np.array([100, 255, 255])

#text display to show what keyboard buttons enable manual override
print("Press 'q' to exit program")
print("Press 's' to manually shoot")

currentAngle_x = 90

# Stabilization tracking variables
stabilization_threshold = 15  # pixels error considered "stable"
stable_frames_required = 5   # number of consecutive stable frames before shooting
stable_frame_count = 0
has_shot = False

kernel = np.ones((3,3), np.uint8)
display_scale = 2  # For displaying at original size

# INCREASED PROCESSING SPEED - process every frame
processing_frame_count = 0
process_every_n_frames = 1  #use 1 for faster detection

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processing_frame_count += 1
        if processing_frame_count % process_every_n_frames != 0:
            # Skip processing this frame, just use for display
            display_frame = cv2.resize(frame, (640, 480))
            cv2.imshow("Tracking", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'): #made ONLY for emergency and testing
                shoot()
            continue

        # Convert to HSV for color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, green_lower, green_upper)

        # Clean up the mask (remove noise)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # Prepare display frame with center lines
        display_frame = cv2.resize(frame, (640, 480))
        h_display, w_display = display_frame.shape[:2]
        cv2.line(display_frame, (w_display // 2, 0), (w_display // 2, h_display), (255, 0, 0), 1)
        cv2.line(display_frame, (0, h_display // 2), (w_display, h_display // 2), (255, 0, 0), 1)

        # Find all the green areas
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

        object_found = False

        if contours:
            # Find largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)

            if area > 300 and area < 10000: #area readjusted to a very large maximum limit due to lighting conditions in the dark room
                # Calculate center
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    # Scale coordinates back to display size
                    cx_display = cx * 2
                    cy_display = cy * 2

                    # Draw on display frame
                    cv2.drawContours(display_frame, [largest_contour * 2], -1, (255, 255, 255), 3)
                    cv2.circle(display_frame, (cx_display, cy_display), 5, (0, 0, 255), -1)

                    # Get frame dimensions for calculations
                    h, w = frame.shape[:2]

                    # Calculate error from center (only X-axis for left-right movement)
                    error_x = cx - (w // 2)

                    step_size = 1.4 # How much to move the servo each adjustment

                    # Check if object is centered (within stabilization threshold)
                    if abs(error_x) <= stabilization_threshold:
                        # Object is centered - count stable frames
                        stable_frame_count += 1
                        # Display stabilization progress using string formatting for the interface
                        cv2.putText(display_frame, "STABILIZING: " + str(stable_frame_count) + "/" + str(stable_frames_required),
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        # Object not centered - move servo and reset stabilization
                        if error_x < 0: # Target is left of center
                            currentAngle_x = min(180, currentAngle_x + step_size)
                        else: # Target is right of center
                            currentAngle_x = max(0, currentAngle_x - step_size)
                        # Reset stabilization counter when moving
                        stable_frame_count = 0
                        has_shot = False

                    # Move only the left-right servo (servo 1)
                    set_servo_angle(1, currentAngle_x)

                    # Check if stabilized and ready to shoot
                    if stable_frame_count >= stable_frames_required and not has_shot:
                        cv2.putText(display_frame, "FIRING!", (w_display//2-50, h_display//2),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        shoot()
                        has_shot = True
                        # If stable frame count counter still increasing high likely hood we missed
                        # attempt to adjust the vertical angle slightly
                        if stable_frame_count >= 15:
                            set_servo_angle(2, 87) #change angle in case target has not been shot down
                            shoot()
                            set_servo_angle(2, 86) #reset angle
                            stable_frame_count = 0  # Reset after shooting


                    # Use string formatting for console output
                    print("Object at: " + str(cx_display) + ", " + str(cy_display) + " | Servo: " + str(currentAngle_x) + " | Stable: " + str(stable_frame_count))
                    object_found = True

        if not object_found:
            print("No object detected")
            # Reset stabilization when object is lost
            stable_frame_count = 0
            has_shot = False

        cv2.imshow("Tracking", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'): # testing/emergency Manual override
            shoot()

except KeyboardInterrupt:
    print('Program interrupted by user')

finally:
    # Cleanup
    set_servo_angle(1, 90)
    motor_off()
    cap.release()
    cv2.destroyAllWindows()
    GPIO.cleanup()
    print("Program ended - GPIO cleanup completed")