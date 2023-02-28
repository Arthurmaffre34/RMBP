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


moteur_ar_d.ChangeDutyCycle(5)
moteur_av_d.ChangeDutyCycle(5)
moteur_av_g.ChangeDutyCycle(5)
moteur_ar_g.ChangeDutyCycle(5)

i = 5
while True:
    moteur_ar_d.ChangeDutyCycle(i)
    moteur_av_d.ChangeDutyCycle(i)
    moteur_av_g.ChangeDutyCycle(i)
    moteur_ar_g.ChangeDutyCycle(i)
    i = int(input("select 5 to 10"))