#include "servo.h"

extern TIM_HandleTypeDef htim3;

void servo_set_angle(uint8_t angle)
{
    if (angle > 180) angle = 180;

    uint16_t pulse = 500 + (angle * 2000 / 180);
    __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_1, pulse);
}