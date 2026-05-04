# 2D-English-Calligraphy-Robot-
The 2D English Calligraphy Robot is an automated drawing system that converts typed text into physical calligraphic writing on paper

What's in This Repo
1- calligraphy_robot.py: Main Python app — GUI + LSTM + G-code generator
2- calligraphy_robot_ForRaspberryPi5.py: Same app configured for Raspberry Pi 5 with serial sending
3- gcode.c / .h: STM32 G-code 
4- parsermain.c / .h: STM32 system init and interrupts
5- motion.c / .h: STM32 Bresenham motion control
6- stepper.c / .h: STM32 stepper motor GPIO control
7- servo.c / .h: STM32 servo PWM control
8- assembly.SLDASM + *.SLDPRT: SolidWorks mechanical design files

 Running the RNN — G-code Generator (What Works Now) (For PC Windows)
This is the only part you need to run right now.
Step 1: Install the one dependency
bashpip install numpy
Step 2: Run the app
bashpython calligraphy_robot.py
Step 3; Use the GUI

Click TRAIN MODEL and wait 10–15 minutes for the LSTM to train
Type any text in the input box
Press GENERATE G-CODE or hit Enter
The G-code appears in the right panel, you can review or copy it

The trained model saves automatically to calligraphy_model.pkl in the same folder. Next time you open the app it loads instantly and you can skip straight to step 2.

 Mechanical Design Files
The files assembly.SLDASM, Base.SLDPRT, Part1–7.SLDPRT, left_bar.SLDPRT, middle_bar.SLDPRT, right_bar.SLDPRT, pen.SLDPRT, pen_Shaft.SLDPRT, and A4 Paper.SLDPRT are the full SolidWorks 3D model of the XY plotter.
Keep all these files in the same folder. Open assembly.SLDASM in SolidWorks and it will automatically load all the part files. Physical construction continues in the summer.

STM32 Firmware — Do Not Run Yet
The files main.c, main.h, gcode.c, gcode.h, motion.c, motion.h, stepper.c, stepper.h, servo.c, and servo.h are the STM32L476 firmware. The code is written and compiles in STM32CubeIDE but has not been tested on physical hardware yet. Do not attempt to run these until the mechanical assembly is complete.
