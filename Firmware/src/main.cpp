#include <Arduino.h>
#include "NeuralNetwork.h"
#include <Arduino.h>
#include <Wire.h>
#include "MLX90640_API.h"
#include "MLX9064X_I2C_Driver.h"
#include <time.h>
#include <string>
#include <stdio.h>
#include <stdlib.h>


// for MLX sensor // --------------------------
uint8_t REFRESH_0_5_HZ = 0b000;  // 0.5Hz
uint8_t  REFRESH_1_HZ = 0b001;  // 1Hz
uint8_t REFRESH_2_HZ = 0b010;   // 2Hz
uint8_t REFRESH_4_HZ = 0b011;   // 4Hz
uint8_t REFRESH_8_HZ = 0b100;   // 8Hz
uint8_t REFRESH_16_HZ = 0b101;  // 16Hz
uint8_t REFRESH_32_HZ = 0b110;  // 32Hz
uint8_t REFRESH_64_HZ = 0b111;  // 64Hz
// default refreshrate
uint8_t refreshrate_u8 = REFRESH_4_HZ; // May be modified based on the above 'refreshrate' configuration
const byte MLX9064x_address = 0x33; //Default 7-bit unshifted address of the MLX9064x // the default I2C address of the sensor
int ambient_temperature_shift = 8; // 5:mlx90641; 8:mlx90640
int reading_type = 2; // 1: one frame contain one subpage. 2: one frame contains two subpages;
float emissivity = 1.0; // the emissivity setting of sensor
#define TA_SHIFT ambient_temperature_shift 
static float mlx90640To[768];
paramsMLX90640 mlx90640;

boolean isConnected() {
    Wire.beginTransmission((uint8_t)MLX9064x_address);
    if (Wire.endTransmission() != 0) {
        return (false);    //Sensor did not ACK
    }
    return (true); //Returns true if the MLX9064x is detected on the I2C bus
}

std::string float_array_to_string(float int_array[], int size_of_array) {
  std::string returnstring = "[";
  for (int temp = 0; temp < size_of_array; temp++){
    returnstring += std::to_string(int_array[temp]);
    returnstring += ",";
  }
  returnstring += "]";
  return returnstring;
}


NeuralNetwork *nn;

void setup()
{
  Serial.begin(115200);
  pinMode(LED_BUILTIN, OUTPUT);
  Serial.println("Hello first");
  // put your setup code here, to run once:

  Wire.begin(17,18, 1000000);  // (SDA_pin,SCL_pin, Baudrate)// i2c setting
  MLX90640_SetRefreshRate((uint8_t)MLX9064x_address, refreshrate_u8);
  Serial.println("Set I2C");
  if (isConnected() == false) {
      Serial.println("MLX9064x not detected at default I2C address. Please check wiring. Freezing.");
      ESP.restart();
  }
  Serial.println("MLX9064x online!");
  int status;
  uint16_t eeMLX9064x[832];
  status = MLX90640_DumpEE(MLX9064x_address, eeMLX9064x);
  if (status != 0) {
      Serial.println("Failed to load system parameters");
  }
  status = MLX90640_ExtractParameters(eeMLX9064x, &mlx90640);
  if (status != 0) {
    Serial.println("Parameter extraction failed");
  }

  nn = new NeuralNetwork();
  Serial.println("Finish Setting");
}


void loop()
{
  for (int pixelNumber = 0; pixelNumber < 768; pixelNumber++) {
        mlx90640To[pixelNumber] = 0.0;
      }
  float ta;
  for (int x = 0 ; x < reading_type ; x++) { //Read both subpages or single subpage
    uint16_t mlx90640Frame[834];
    int status = MLX90640_GetFrameData(MLX9064x_address, mlx90640Frame);
    if (status < 0) {
        Serial.print("GetFrame Error: ");
        Serial.println(status);
        x--;
        continue;
    }
    float vdd = MLX90640_GetVdd(mlx90640Frame, &mlx90640);
    float Ta = MLX90640_GetTa(mlx90640Frame, &mlx90640);
    float tr = Ta - ambient_temperature_shift; //Reflected temperature based on the sensor ambient temperature
    ta = Ta - ambient_temperature_shift;
    MLX90640_CalculateTo(mlx90640Frame, &mlx90640, emissivity, tr, mlx90640To);
  }

  float *inputBuffer = nn->getInputBuffer();
  for (int i = 0; i < 768; i++) {
    inputBuffer[i] = mlx90640To[i];
  }

  // Get the output vector from the model
  float start_inference = millis();
  float *outputVector = nn->predict();
  float inference_delay = millis() - start_inference;

  // std::string message = "{";  // the message to be sent
  // std::string timestamp = " \"Timestamp\": \"";
  // timestamp += "none\"";
  // message += timestamp;
  // message += ", \"loc_ts\":";
  // message += std::to_string(millis());
  // message += ", \"AT\":";
  // message += std::to_string(ta);
  // message += ", \"data\":";
  // std::string data = float_array_to_string(mlx90640To, 768);
  // message += data;
  // message += "}";

  // Serial.println(message.c_str());

  // Print the output vector
  Serial.print("Output vector: [");
  for (int i = 0; i < 63; ++i) {
    Serial.print(outputVector[i]);
    if (i < 62) {
      Serial.print(", ");
    }
  }
  Serial.println("]");

  std::string message = "Inference time delay (millis): ";
  message += std::to_string(inference_delay);
  Serial.println(message.c_str());
  // Serial.println("Finish Setting");
  // delay(1000);
}