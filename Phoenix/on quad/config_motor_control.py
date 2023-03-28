import RPi.GPIO as GPIO
import time

# Set GPIO numbering mode
GPIO.setmode(GPIO.BCM)

# Setup motor gpio output
GPIO.setup(26,GPIO.OUT)
GPIO.setup(19,GPIO.OUT)
GPIO.setup(13,GPIO.OUT)
GPIO.setup(6,GPIO.OUT)

moteur_ar_d = GPIO.PWM(26, 50) # Note 26 is pin, 50 = 50hz pulse
moteur_av_d = GPIO.PWM(19, 50)
moteur_av_g = GPIO.PWM(13, 50)
moteur_ar_g = GPIO.PWM(6, 50)

print("déconnecter les moteurs set max")
#config max and low power
moteur_ar_d.start(10)
moteur_av_d.start(10)
moteur_av_g.start(10)
moteur_ar_g.start(10)
input("connecter les moteurs")
time.sleep(3)
moteur_ar_d.ChangeDutyCycle(5)
moteur_av_d.ChangeDutyCycle(5)
moteur_av_g.ChangeDutyCycle(5)
moteur_ar_g.ChangeDutyCycle(5)
print("set to 0")
time.sleep(1.5)
GPIO.cleanup()