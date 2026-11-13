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

extern bool PSU_isStable;	// If the power supply is stable
extern bool PSU_isOn;		// If the power supply is switched on

void PSU_GPIO_Init(void);
void PSU_SwitchOn(void);
void PSU_SwitchOff(void);
void PSU_TogglePower(void);
bool PSU_CheckPowerGood(void);

#define PSU_DC_OK_GPIO_Port GPIOC
#define PSU_DC_OK_Pin GPIO_PIN_10

#define PSU_REMOTE_CONTROL_GPIO_Port GPIOC
#define PSU_REMOTE_CONTROL_Pin GPIO_PIN_11

#endif // __POWER_SUPPLY_H
