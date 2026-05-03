#include "main.h"
#include "gcode.h"
#include "motion.h"

#include <string.h>


UART_HandleTypeDef huart1;
TIM_HandleTypeDef htim3;
TIM_HandleTypeDef htim6;

uint8_t rx_char;
char buffer[128];
int idx = 0;	

void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
    if (rx_char == '\n')
    {
        buffer[idx] = '\0';
        process_gcode(buffer);
        idx = 0;
    }
    else if (idx < sizeof(buffer) - 1)
    {
        buffer[idx++] = rx_char;
    }
	else {
		/*Do Nothing*/
	}

    HAL_UART_Receive_IT(&huart1, &rx_char, 1);
}


void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim)
{
    if (htim->Instance == TIM6)
    {
        motion_execute_step();
    }
}

int main(void)
{	
    HAL_Init();
    SystemClock_Config();

    MX_GPIO_Init();
    MX_USART1_UART_Init();
    MX_TIM3_Init();  // Servo
    MX_TIM6_Init();  // Stepper ISR

    HAL_TIM_PWM_Start(&htim3, TIM_CHANNEL_1);
	HAL_TIM_Base_Start_IT(&htim6);


    HAL_UART_Receive_IT(&huart1, &rx_char, 1);

    while (1)
    {
    }
}