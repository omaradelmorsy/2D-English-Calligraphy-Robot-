// gcode.c
#include "motion.h"
#include <string.h>
#include <stdlib.h>

#include "gcode.h"
#include "motion.h"
#include "servo.h

#define STEPS_PER_MM 80

void process_gcode(char *line)
{
    if (strncmp(line, "G0", 2) == 0 || strncmp(line, "G1", 2) == 0)
    {
        float x=0, y=0;

        char *token = strtok(line, " ");
        while (token)
        {
            if (token[0]=='X') x = atof(&token[1]);
            if (token[0]=='Y') y = atof(&token[1]);
            token = strtok(NULL, " ");
        }

        motion_enqueue((int)(x*STEPS_PER_MM), (int)(y*STEPS_PER_MM));
    }

    else if (strncmp(line, "M3", 2) == 0)
    {
         char *s_ptr = strchr(line, 'S');
        if (s_ptr)
        {
            int angle = atoi(s_ptr + 1);
            servo_set_angle(angle);
        }
    }
}