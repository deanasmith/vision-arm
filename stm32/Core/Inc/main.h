/**
  ******************************************************************************
  * @file           : main.h
  * @brief          : Header for main.c file.
  *                   This file contains the common defines of the application.
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __MAIN_H
#define __MAIN_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32f0xx_hal.h"

/* Private includes ----------------------------------------------------------*/

/* Exported types ------------------------------------------------------------*/

/* Exported constants --------------------------------------------------------*/

/* Exported macro ------------------------------------------------------------*/

/* Exported functions prototypes ---------------------------------------------*/
void Error_Handler(void);

/* Private defines -----------------------------------------------------------*/
// Default timer values
#define TIMER_PSC 14U
#define TIMER_ARR 64000U

//Pin definitions
#define NCS_MEMS_SPI_Pin GPIO_PIN_0
#define NCS_MEMS_SPI_GPIO_Port GPIOC
#define MEMS_INT1_Pin GPIO_PIN_1
#define MEMS_INT1_GPIO_Port GPIOC
#define MEMS_INT2_Pin GPIO_PIN_2
#define MEMS_INT2_GPIO_Port GPIOC
#define B1_Pin GPIO_PIN_0
#define B1_GPIO_Port GPIOA
#define EXT_RESET_Pin GPIO_PIN_5
#define EXT_RESET_GPIO_Port GPIOC
#define I2C2_SCL_Pin GPIO_PIN_10
#define I2C2_SCL_GPIO_Port GPIOB
#define I2C2_SDA_Pin GPIO_PIN_11
#define I2C2_SDA_GPIO_Port GPIOB
#define SPI2_SCK_Pin GPIO_PIN_13
#define SPI2_SCK_GPIO_Port GPIOB
#define SPI2_MISO_Pin GPIO_PIN_14
#define SPI2_MISO_GPIO_Port GPIOB
#define SPI2_MOSI_Pin GPIO_PIN_15
#define SPI2_MOSI_GPIO_Port GPIOB
#define LD3_Pin GPIO_PIN_6
#define LD3_GPIO_Port GPIOC
#define LD6_Pin GPIO_PIN_7
#define LD6_GPIO_Port GPIOC
#define LD4_Pin GPIO_PIN_8
#define LD4_GPIO_Port GPIOC
#define LD5_Pin GPIO_PIN_9
#define LD5_GPIO_Port GPIOC
#define USBF4_DM_Pin GPIO_PIN_11
#define USBF4_DM_GPIO_Port GPIOA
#define USBF4_DP_Pin GPIO_PIN_12
#define USBF4_DP_GPIO_Port GPIOA
#define SWDIO_Pin GPIO_PIN_13
#define SWDIO_GPIO_Port GPIOA
#define SWCLK_Pin GPIO_PIN_14
#define SWCLK_GPIO_Port GPIOA

// USART3
#define USART3_GPIO_Port GPIOB
#define USART3_TX_Pin GPIO_PIN_10
#define USART3_RX_Pin GPIO_PIN_11

// TIM1
#define TIM1_GPIO_Port GPIOA
#define TIM1_CH1_Pin GPIO_PIN_8
#define TIM1_CH2_Pin GPIO_PIN_9
#define TIM1_CH3_Pin GPIO_PIN_10
#define TIM1_CH4_Pin GPIO_PIN_11	// TIM1_CH4 is assigned to the USER USB port on our STM32, so it can't be used

// TIM2
#define TIM2_CH1_GPIO_Port GPIOA
#define TIM2_CH1_Pin GPIO_PIN_15

#define TIM2_GPIO_Port GPIOB
#define TIM2_CH2_Pin GPIO_PIN_3
#define TIM2_CH3_Pin GPIO_PIN_10
#define TIM2_CH4_Pin GPIO_PIN_11

// TIM3
#define TIM3_GPIO_Port GPIOC
#define TIM3_CH1_Pin GPIO_PIN_6
#define TIM3_CH2_Pin GPIO_PIN_7
#define TIM3_CH3_Pin GPIO_PIN_8
#define TIM3_CH4_Pin GPIO_PIN_9

// TIM14
#define TIM14_GPIO_Port GPIOA
#define TIM14_CH1_Pin GPIO_PIN_4

// TIM15
#define TIM15_GPIO_Port GPIOB
#define TIM15_CH1_Pin GPIO_PIN_14
#define TIM15_CH2_Pin GPIO_PIN_15

// TIM16
#define TIM16_GPIO_Port GPIOB
#define TIM16_CH1_Pin GPIO_PIN_8

// TIM17
#define TIM17_GPIO_Port GPIOB
#define TIM17_CH1_Pin GPIO_PIN_9


#ifdef __cplusplus
}
#endif

#endif /* __MAIN_H */
