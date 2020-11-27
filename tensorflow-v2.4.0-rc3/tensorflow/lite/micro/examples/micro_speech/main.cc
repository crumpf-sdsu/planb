/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

//#include "tensorflow/lite/micro/examples/micro_speech/main_functions.h"

// This is the default main used on systems that have the standard C entry
// point. Other devices (for example FreeRTOS or ESP32) that have different
// requirements for entry code (like an app_main function) should specialize
// this main.cc file in a target-specific subfolder.
//int main(int argc, char* argv[]) {
//  setup();
//  while (true) {
//    loop();
//  }
//}


/* mbed Microcontroller Library
 * Copyright (c) 2019 ARM Limited
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mbed.h"
#include "platform/mbed_thread.h"
#include "DHT22.h"
#include "LCD_DISCO_F746NG.h"
#include "main_functions.h"

// Blinking rate in milliseconds
#define INIT_SLEEP_RATE_MS 2000
#define SLEEP_RATE 86400000 // 1 hour
#define SLEEP_RATE_DEBUG 10000 // 10s
#define STACK_SIZE 7

void push(int top, float element);

DHT22 sensor(PG_6);
LCD_DISCO_F746NG lcd1;

float past_temps[STACK_SIZE];
int front = 0;
int rear = -1;
int itemCount = 0;

int peek() {
    return past_temps[front];
}

bool isEmpty() {
    return itemCount == 0;
}

bool isFull() {
    return itemCount == STACK_SIZE;
}

int size() {
    return itemCount;
}

int removeData() {
   int data = past_temps[front++];
	
   if(front == STACK_SIZE) {
      front = 0;
   }
	
   itemCount--;
   return data;
}

void insert(float data) {

    if(isFull()) {
        removeData();
    }
	
    if(rear == STACK_SIZE-1) {
         rear = -1;            
    }       

    past_temps[++rear] = data;
    itemCount++;
}


int main()
{
    // Variable initialization
    int err;
    char str[80];
    int hours_counted = 0;
    float pt = 99.0;

    // Initialize the temperature stack
    float seed = 69;
    for( int i=0; i<STACK_SIZE; i++){
        seed++;
        insert(seed);
    }

    // Initialise the digital pin LED1 as an output
    DigitalOut led(LED1);

    // LCD Initialization
    lcd1.SetFont(&Font12);
    lcd1.Clear(LCD_COLOR_BLUE);
    lcd1.SetBackColor(LCD_COLOR_BLUE);
    lcd1.SetTextColor(LCD_COLOR_WHITE);
    lcd1.DisplayStringAt(0, LINE(1), (uint8_t *)"Mr. Rumpf's PlanB", CENTER_MODE);
    lcd1.DisplayStringAt(0, LINE(2), (uint8_t *)"Weather Prediction Using TensorFlowLite", CENTER_MODE);
    lcd1.DisplayStringAt(0, LINE(3), (uint8_t *)"NOAA Dataset", CENTER_MODE);

    // Hardware initialization sleep for DHT22 
    wait_ms(INIT_SLEEP_RATE_MS);

    // TensorflowLite initialization
    //setup();

    // Read and predict temperature 1x every hour
    while(1){
        // Incriment number of hours (uptime)
        hours_counted++;
        sprintf(str, "RUN #%d", hours_counted);
        lcd1.DisplayStringAt(0, LINE(5), (uint8_t *)str, CENTER_MODE);
        lcd1.DrawHLine(110, 70, 250);

        // Flip the LED
        led = !led;
    
        // Read, convert sensor data and push to queue data structure
        err = sensor.readData();
        float t = sensor.getTemperatureF();
        if (err == 0) {
            sprintf(str, "Current Temperature : %4.2f F", t);
        } else {
            sprintf(str, "Error %i", hours_counted, err);
        }
        lcd1.DisplayStringAt(0, LINE(7), (uint8_t *)str, CENTER_MODE);
        insert(t);

        // Print the past 7 temps (model input) to LCD
        char temps[40] = "";
        for(int i=0; i<STACK_SIZE; ++i) {
            sprintf(temps, "%s %4.1f", temps, past_temps[i]);
        }
        sprintf(str, "Last 7 Temps: %s", temps);
        lcd1.DisplayStringAt(0, LINE(8), (uint8_t *)str, CENTER_MODE);

        // Predict weather for the next 1 hour
        //pt = loop(past_temps);
        pt++;
        sprintf(str, "PREDICTED TEMPERATURE FOR NEXT 1 HOUR : %4.2f F", pt);
        lcd1.DisplayStringAt(0, LINE(11), (uint8_t *)str, CENTER_MODE);
        
        // Print some debug insights
        printf("\r\nHour #: %d\r\n", hours_counted);
        for( int i=0; i < STACK_SIZE; i++) {
            printf("%f,", past_temps[i]);
        }
        printf("\r\n");

        // Sleep for an hour
        wait_ms(SLEEP_RATE_DEBUG);
    }
}
