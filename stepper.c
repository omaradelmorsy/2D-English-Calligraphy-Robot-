// stepper.c
#include "main.h"


void step_x(int dir)
{
    HAL_GPIO_WritePin(GPIOA, GPIO_PIN_0, dir);
    HAL_GPIO_WritePin(GPIOA, GPIO_PIN_1, 1);
    HAL_GPIO_WritePin(GPIOA, GPIO_PIN_1, 0);
}

void step_y(int dir)
{
    HAL_GPIO_WritePin(GPIOA, GPIO_PIN_2, dir);
    HAL_GPIO_WritePin(GPIOA, GPIO_PIN_3, 1);
    HAL_GPIO_WritePin(GPIOA, GPIO_PIN_3, 0);
}