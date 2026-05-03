// motion.h
#ifndef MOTION_H
#define MOTION_H


#include "main.h"


typedef struct {
    int x;
    int y;
} MotionBlock;

void motion_enqueue(int x, int y, int feed);
void motion_execute_step(void);




#endif