/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
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
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "stm32f0xx_hal.h"
#include "stm32f0xx_hal_gpio.h"
#include "stm32f0xx_hal_gpio_ex.h"
#include "stm32f0xx_hal_rcc.h"
#include "stm32f0xx_hal_rcc_ex.h"
#include "stm32f0xx_hal_uart.h"
#include "stm32f0xx_hal_tim.h"

/* Private includes ----------------------------------------------------------*/

/* Private typedef -----------------------------------------------------------*/

/* Private define ------------------------------------------------------------*/

/* Private macro -------------------------------------------------------------*/

/* Private variables ---------------------------------------------------------*/
UART_HandleTypeDef uart3_handle = {0};
TIM_HandleTypeDef tim1_handle = {0};
TIM_HandleTypeDef tim3_handle = {0};

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);

// GPIO initialization
static void MX_GPIO_Init(void);
static void GPIO_Init(void);
static void USART3_GPIO_Init(void);
static void TIM1_GPIO_Init(void);
static void TIM3_GPIO_Init(void);

// Peripheral initialization
static void USART_Init(void);
static void TIM1_Init(void);
static void TIM3_Init(void);
static void configure_timers(void);

// Helper functions
static TIM_Base_InitTypeDef get_timer_init_struct(void);

/* Private user code ---------------------------------------------------------*/

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{
  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* Configure the system clock */
  SystemClock_Config();

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  GPIO_Init();
  USART_Init();
  TIM1_Init();
  TIM3_Init();
  configure_timers();

  HAL_TIM_PWM_Start(&tim1_handle, TIM_CHANNEL_1);
  HAL_TIM_PWM_Start(&tim1_handle, TIM_CHANNEL_2);
  HAL_TIM_PWM_Start(&tim1_handle, TIM_CHANNEL_3);

  HAL_TIM_PWM_Start(&tim3_handle, TIM_CHANNEL_1);
  HAL_TIM_PWM_Start(&tim3_handle, TIM_CHANNEL_2);
  HAL_TIM_PWM_Start(&tim3_handle, TIM_CHANNEL_3);
  HAL_TIM_PWM_Start(&tim3_handle, TIM_CHANNEL_4);

  // TODO: Send the ARR to the computer vision code on startup to let it
  // calculate duty cycle CCR values instead for faster processing
  uint8_t tx_buffer[100] = "Hello World!\n";
  uint8_t rx_buffer[100];
  /* Infinite loop */
  while (1)
  {
    HAL_UART_Receive(&uart3_handle, rx_buffer, 100, 1000000U);
    HAL_UART_Transmit(&uart3_handle, tx_buffer, 14U, 1000000U);
    HAL_Delay(1000U);
  }
}


/**
  * @brief Initialize GPIO
  * @param None
  * @retval None
  */
static void GPIO_Init(void)
{
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();
  __HAL_RCC_GPIOC_CLK_ENABLE();

  USART3_GPIO_Init();
  TIM1_GPIO_Init();
  TIM3_GPIO_Init();
}


/**
  * @brief Initialize USART GPIO pins
  * @param None
  * @retval None
  */
static void USART3_GPIO_Init(void)
{
  GPIO_InitTypeDef uart_init_struct = {0};

  // Initialize UART TX pin
  uart_init_struct.Pin       = USART3_TX_Pin;
  uart_init_struct.Mode      = GPIO_MODE_AF_PP;
  uart_init_struct.Pull      = GPIO_NOPULL;
  uart_init_struct.Speed     = GPIO_SPEED_FREQ_LOW;
  uart_init_struct.Alternate = GPIO_AF4_USART3;

  HAL_GPIO_Init(USART3_GPIO_Port, &uart_init_struct);

  // Initialize UART RX pin
  uart_init_struct.Pin       = USART3_RX_Pin;
  uart_init_struct.Mode      = GPIO_MODE_AF_OD;
  uart_init_struct.Pull      = GPIO_NOPULL;
  uart_init_struct.Speed     = GPIO_SPEED_FREQ_LOW;
  uart_init_struct.Alternate = GPIO_AF4_USART3;

  HAL_GPIO_Init(USART3_GPIO_Port, &uart_init_struct);
}


/**
 * @brief Initailize TIM1 GPIO pins
 * @param None
 * @retval None
 */
static void TIM1_GPIO_Init(void)
{
  GPIO_InitTypeDef tim1_init_struct = {0};

  tim1_init_struct.Pin =
    TIM1_CH1_Pin |
    TIM1_CH2_Pin |
    TIM1_CH3_Pin |
    TIM1_CH4_Pin;

  tim1_init_struct.Mode      = GPIO_MODE_AF_PP;
  tim1_init_struct.Pull      = GPIO_NOPULL;
  tim1_init_struct.Speed     = GPIO_SPEED_FREQ_LOW;
  tim1_init_struct.Alternate = GPIO_AF2_TIM1;

  HAL_GPIO_Init(TIM1_GPIO_Port, &tim1_init_struct);
}


/**
 * @brief Initailize TIM3 GPIO pins
 * @param None
 * @retval None
 */
static void TIM3_GPIO_Init(void)
{
  GPIO_InitTypeDef tim3_init_struct = {0};

  tim3_init_struct.Pin =
    TIM3_CH1_Pin |
    TIM3_CH2_Pin |
    TIM3_CH3_Pin |
    TIM3_CH4_Pin;

  tim3_init_struct.Mode      = GPIO_MODE_AF_PP;
  tim3_init_struct.Pull      = GPIO_NOPULL;
  tim3_init_struct.Speed     = GPIO_SPEED_FREQ_LOW;
  tim3_init_struct.Alternate = GPIO_AF0_TIM3;

  HAL_GPIO_Init(TIM3_GPIO_Port, &tim3_init_struct);
}


/**
  * @brief Initialize USART peripheral in UART mode
  * @param None
  * @retval None
  */
static void USART_Init(void)
{
  __HAL_RCC_USART3_CLK_ENABLE();

  UART_InitTypeDef uart_init_struct = {0};
  uart_init_struct.BaudRate   = 115200U;
  uart_init_struct.WordLength = UART_WORDLENGTH_8B;
  uart_init_struct.StopBits   = UART_STOPBITS_1;
  uart_init_struct.Parity     = UART_PARITY_NONE;
  uart_init_struct.Mode       = UART_MODE_TX_RX;

  // Setup uart3_handle struct
  uart3_handle.Instance = USART3;
  uart3_handle.Init     = uart_init_struct;

  HAL_UART_Init(&uart3_handle);
}


/**
 * @brief Initialize TIM1 peripheral in PWM output mode
 * @param None
 * @retval None
 */
static void TIM1_Init(void)
{
  __HAL_RCC_TIM1_CLK_ENABLE();

  tim1_handle.Instance = TIM1;
  tim1_handle.Init     = get_timer_init_struct();
  tim1_handle.Channel  =
    TIM_CHANNEL_1 |
    TIM_CHANNEL_2 |
    TIM_CHANNEL_3 |
    TIM_CHANNEL_4;

  HAL_TIM_PWM_Init(&tim1_handle);
}


/**
 * @brief Initialize TIM3 peripheral in PWM output mode
 * @param None
 * @retval None
 */
static void TIM3_Init(void)
{
  __HAL_RCC_TIM3_CLK_ENABLE();

  tim3_handle.Instance = TIM3;
  tim3_handle.Init     = get_timer_init_struct();
  tim3_handle.Channel  =
    TIM_CHANNEL_1 |
    TIM_CHANNEL_2 |
    TIM_CHANNEL_3 |
    TIM_CHANNEL_4;

  HAL_TIM_PWM_Init(&tim3_handle);
}


/**
 * @brief Returns a timer init struct for baseline PWM setup
 * @param None
 * @retval A timer init struct for baseline PWM setup
 */
static TIM_Base_InitTypeDef get_timer_init_struct(void)
{
  // TODO: Set the prescaler and ARR values dynamically
  // HAL_RCC_GetPCLK1Freq();

  // Set the timer to 50 Hz with an 48MHz PCLK
  // PCLK = 48 MHz
  // f_target = 50 Hz
  // PSC = 14   // This is the minimum for ARR to be within the 16-bit register
  // ARR = PCLK/((PSC + 1)*f_target)  // ARR == 64000
  TIM_Base_InitTypeDef timer_init_struct = {0};
  timer_init_struct.Prescaler         = TIMER_PSC;
  timer_init_struct.CounterMode       = TIM_COUNTERMODE_UP;
  timer_init_struct.Period            = TIMER_ARR;
  timer_init_struct.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_ENABLE;

  return timer_init_struct;
}


/**
 * @brief Configure timer peripherals for default PWM settings
 * @param None
 * @retval None
 */
static void configure_timers(void)
{
  TIM_OC_InitTypeDef timer_config = {0};

  timer_config.OCMode     = TIM_OCMODE_PWM1;
  timer_config.Pulse      = TIMER_DEFAULT_CRR;
  timer_config.OCPolarity = TIM_OCPOLARITY_HIGH;

  HAL_TIM_PWM_ConfigChannel(&tim1_handle, &timer_config, TIM_CHANNEL_1);
  HAL_TIM_PWM_ConfigChannel(&tim1_handle, &timer_config, TIM_CHANNEL_2);
  HAL_TIM_PWM_ConfigChannel(&tim1_handle, &timer_config, TIM_CHANNEL_3);

  HAL_TIM_PWM_ConfigChannel(&tim3_handle, &timer_config, TIM_CHANNEL_1);
  HAL_TIM_PWM_ConfigChannel(&tim3_handle, &timer_config, TIM_CHANNEL_2);
  HAL_TIM_PWM_ConfigChannel(&tim3_handle, &timer_config, TIM_CHANNEL_3);
  HAL_TIM_PWM_ConfigChannel(&tim3_handle, &timer_config, TIM_CHANNEL_4);
}


/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOC, NCS_MEMS_SPI_Pin|EXT_RESET_Pin|LD3_Pin|LD6_Pin
                          |LD4_Pin|LD5_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pins : NCS_MEMS_SPI_Pin EXT_RESET_Pin LD3_Pin LD6_Pin
                           LD4_Pin LD5_Pin */
  GPIO_InitStruct.Pin = NCS_MEMS_SPI_Pin|EXT_RESET_Pin|LD3_Pin|LD6_Pin
                          |LD4_Pin|LD5_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);

  /*Configure GPIO pins : MEMS_INT1_Pin MEMS_INT2_Pin */
  GPIO_InitStruct.Pin = MEMS_INT1_Pin|MEMS_INT2_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_EVT_RISING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);

  /*Configure GPIO pin : B1_Pin */
  GPIO_InitStruct.Pin = B1_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_EVT_RISING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(B1_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pins : PA2 PA3 PA6 PA7 */
  GPIO_InitStruct.Pin = GPIO_PIN_2|GPIO_PIN_3|GPIO_PIN_6|GPIO_PIN_7;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  GPIO_InitStruct.Alternate = GPIO_AF3_TSC;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  /*Configure GPIO pins : PB0 PB1 */
  GPIO_InitStruct.Pin = GPIO_PIN_0|GPIO_PIN_1;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  GPIO_InitStruct.Alternate = GPIO_AF3_TSC;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

  /*Configure GPIO pins : I2C2_SCL_Pin I2C2_SDA_Pin */
  GPIO_InitStruct.Pin = I2C2_SCL_Pin|I2C2_SDA_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_OD;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
  GPIO_InitStruct.Alternate = GPIO_AF1_I2C2;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

  /*Configure GPIO pins : SPI2_SCK_Pin SPI2_MISO_Pin SPI2_MOSI_Pin */
  GPIO_InitStruct.Pin = SPI2_SCK_Pin|SPI2_MISO_Pin|SPI2_MOSI_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
  GPIO_InitStruct.Alternate = GPIO_AF0_SPI2;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

}


/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLMUL = RCC_PLL_MUL6;
  RCC_OscInitStruct.PLL.PREDIV = RCC_PREDIV_DIV1;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_1) != HAL_OK)
  {
    Error_Handler();
  }
}


/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}

#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
        // printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
