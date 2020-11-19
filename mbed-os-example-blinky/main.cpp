/* mbed Microcontroller Library
 * Copyright (c) 2019 ARM Limited
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mbed.h"
#include "platform/mbed_thread.h"
#include "DHT22.h"
#include "LCD_DISCO_F746NG.h"

// Blinking rate in milliseconds
#define SLEEP_RATE_MS 2000

DHT22 sensor(PG_6);
LCD_DISCO_F746NG lcd;

int main()
{
    int err;
    char str[80];

    // Initialise the digital pin LED1 as an output
    DigitalOut led(LED1);

    lcd.SetFont(&Font16);
    lcd.Clear(LCD_COLOR_BLUE);
    lcd.SetBackColor(LCD_COLOR_BLUE);1
    lcd.SetTextColor(LCD_COLOR_WHITE);
    lcd.DisplayStringAt(0, LINE(1), (uint8_t *)"Mr. Rumpf's PlanB", CENTER_MODE);
    lcd.DisplayStringAt(0, LINE(2), (uint8_t *)"Weather Prediction Using TensorFlowLite", CENTER_MODE);
    lcd.DisplayStringAt(0, LINE(3), (uint8_t *)"NOAA Dataset", CENTER_MODE);
    lcd.DrawHLine(110, 70, 250);

    ThisThread::sleep_for(5000);
        
    while (true) {
        
        led = !led;
        ThisThread::sleep_for(SLEEP_RATE_MS);
        err = sensor.readData();
        if (err == 0) {
            sprintf(str, "Current Temperature : %4.2f F",sensor.getTemperatureF());
        } else {
            sprintf(str, "Err %i",err);
        }
        lcd.DisplayStringAt(0, LINE(6), (uint8_t *)str, CENTER_MODE);

    }
}
