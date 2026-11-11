/**
  ******************************************************************************
  * @file           : power_supply.h
  * @brief          : Header for power_supply.c file.
  *                   This file contains the common defines of the application.
  ******************************************************************************
  */
#ifndef __POWER_SUPPLY_H
#define __POWER_SUPPLY_H

#include <stdbool.h>
#include "stm32f0xx_hal.h"

void PSU_GPIO_Init(void);

#define PSU_DC_OK_GPIO_Port GPIOC
#define PSU_DC_OK_Pin GPIO_PIN_10

#define PSU_REMOTE_CONTROL_GPIO_Port GPIOC
#define PSU_REMOTE_CONTROL_Pin GPIO_PIN_11

#endif // __POWER_SUPPLY_H
