###############################################################################
# Hello Zerynth
#
# Created by Zerynth Team 2015 CC
# Authors: G. Baldi, D. Mazzei
###############################################################################

# import the streams module, it is needed to send data around
import streams

# open the default serial port, the output will be visible in the serial console
streams.serial()  

# loop forever
while True:
    print("Hello Zerynth!")   # print automatically knows where to print!
    sleep(1000)