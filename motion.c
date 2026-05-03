// motion.c
#include "motion.h"

#define QUEUE_SIZE 16

static MotionBlock queue[QUEUE_SIZE];
static int head = 0, tail = 0;
 
static int current_x = 0;
static int current_y = 0;

static MotionBlock current;
static int steps, dx, dy, sx, sy, err;

static int running = 0;		

void motion_enqueue(int x, int y)
{
    int next = (head + 1) % QUEUE_SIZE;
    if (next == tail) return; 	

    queue[head].x = x;
    queue[head].y = y;
	
    head = next;
}


static void motion_start_block(MotionBlock b)
{
    current = b;

    dx = abs(b.x - current_x);		
    dy = abs(b.y - current_y);

    sx = (current_x < b.x) ? 1 : -1;
    sy = (current_y < b.y) ? 1 : -1;

    err = dx - dy;
}

void motion_execute_step(void)
{
    if (head == tail) return;

    if (!running)
    {
        motion_start_block(queue[tail]);
        running = 1;
    }

    int e2 = 2 * err;

    if (e2 > -dy)
    {
        err -= dy;
        current_x += sx;
        step_x(sx > 0);
    }

    if (e2 < dx)
    {
        err += dx;
        current_y += sy;
        step_y(sy > 0);
    }

    if (current_x == current.x && current_y == current.y)
    {
        tail = (tail + 1) % QUEUE_SIZE;
        running = 0;
    }
}