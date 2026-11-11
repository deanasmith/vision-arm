/**
  ******************************************************************************
  * @file           : power_supply.c
  * @brief          : Controls/Receives signals to/from the power supply
  ******************************************************************************
  */

#include "power_supply.h"
#include "stm32f0xx_hal.h"
#include "stm32f0xx_hal_gpio.h"

/**
 * @brief Initializes GPIO pins for power supply signals
 * @param None
 * @retval None
 */
void PSU_GPIO_Init(void)
{
  // Initialize PSU DC OK pin
  GPIO_InitTypeDef psu_init_struct = {0};
  psu_init_struct.Pin       = PSU_DC_OK_Pin;
  psu_init_struct.Mode      = GPIO_MODE_INPUT;
  psu_init_struct.Pull      = GPIO_NOPULL;
  psu_init_struct.Speed     = GPIO_SPEED_FREQ_LOW;

  HAL_GPIO_Init(PSU_DC_OK_GPIO_Port, &psu_init_struct);

  // Initialize PSU remote control pin
  psu_init_struct = (GPIO_InitTypeDef){0};
  psu_init_struct.Pin       = PSU_REMOTE_CONTROL_Pin;
  psu_init_struct.Mode      = GPIO_MODE_OUTPUT_PP;
  psu_init_struct.Pull      = GPIO_NOPULL;
  psu_init_struct.Speed     = GPIO_SPEED_FREQ_LOW;

  HAL_GPIO_Init(PSU_REMOTE_CONTROL_GPIO_Port, &psu_init_struct);
}
