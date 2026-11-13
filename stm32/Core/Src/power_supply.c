/**
  ******************************************************************************
  * @file           : power_supply.c
  * @brief          : Controls/Receives signals to/from the power supply
  ******************************************************************************
  */

#include "power_supply.h"
#include "stm32f0xx_hal.h"
#include "stm32f0xx_hal_gpio.h"

bool PSU_isStable = false;
bool PSU_isOn = false;

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
  PSU_SwitchOff();
}

/**
 * @brief Toggles the power supply on
 * @param None
 * @retval none
 */
void PSU_SwitchOn(void)
{
  // The power supply has inverted logic
  // The PSU is switched on by a low signal
  HAL_GPIO_WritePin(PSU_REMOTE_CONTROL_GPIO_Port,
                    PSU_REMOTE_CONTROL_Pin,
                    GPIO_PIN_RESET);

  PSU_isOn = true;
}

/**
 * @brief Toggles the power supply off
 * @param None
 * @retval none
 */
void PSU_SwitchOff(void)
{
  // The power supply has inverted logic
  // The PSU is switched off by a high signal
  HAL_GPIO_WritePin(PSU_REMOTE_CONTROL_GPIO_Port,
                    PSU_REMOTE_CONTROL_Pin,
                    GPIO_PIN_SET);

  PSU_isOn = false;
}

/**
 * @brief Toggles the power supply on or off
 * @param None
 * @retval none
 */
void PSU_TogglePower(void)
{
  if (PSU_isOn)
    PSU_SwitchOff();
  else
    PSU_SwitchOn();
}

/**
 * @brief Checks if the power supply is stable (DC OK)
 * @param None
 * @retval true if the power supply is stable; otherwise false
 */
bool PSU_CheckPowerGood(void)
{
  // The power supply has inverted logic
  // If the DC_OK pin is low, the PSU is enabled, which pulls DC_OK to ground
  PSU_isStable = !(bool)HAL_GPIO_ReadPin(PSU_DC_OK_GPIO_Port, PSU_DC_OK_Pin);
  return PSU_isStable;
}
