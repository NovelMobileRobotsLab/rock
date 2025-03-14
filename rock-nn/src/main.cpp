#include <Arduino.h>
// #include <mnist_model.h>
// #include <sin_model.h>
// #include <sin_model_512.h>
#include <piecewise_model_128.h>
#include <all_ops_resolver.h>


#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#define TAG "main"


tflite::ErrorReporter *error_reporter = nullptr;
const tflite::Model *model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input = nullptr;
// An area of memory to use for input, output, and intermediate arrays.
const int kTensorArenaSize = 35 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];


void printModelInfo(){
  input = interpreter->input(0);
  ESP_LOGI(TAG, "Input Shape");
  for (int i = 0; i < input->dims->size; i++)
  {
    ESP_LOGI(TAG, "%d", input->dims->data[i]);
  }

  ESP_LOGI(TAG, "Input Type: %s", TfLiteTypeGetName(input->type));
  ESP_LOGI(TAG, "Output Shape");

  TfLiteTensor *output = interpreter->output(0);
  for (int i = 0; i < output->dims->size; i++)
  {
    ESP_LOGI(TAG, "%d", output->dims->data[i]);
  }
  ESP_LOGI(TAG, "Output Type: %s", TfLiteTypeGetName(output->type));

  ESP_LOGI(TAG, "Arena Size:%d bytes of memory", interpreter->arena_used_bytes());
}



void setup() {
    Serial.begin(115200);
    pinMode(LED_BUILTIN, OUTPUT);
    digitalWrite(LED_BUILTIN, HIGH);
    delay(2000);
    digitalWrite(LED_BUILTIN, LOW);    

    static tflite::MicroErrorReporter micro_error_reporter;
    error_reporter = &micro_error_reporter;
    // model = tflite::GetModel(mnist_model);
    // model = tflite::GetModel(sin_model);
    // model = tflite::GetModel(sin_model_512);
    model = tflite::GetModel(piecewise_model_128);
    if (model->version() != TFLITE_SCHEMA_VERSION){
        ESP_LOGE(TAG, "Model provided is schema version %d not equal to supported version %d.", model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }
    CREATE_ALL_OPS_RESOLVER(op_resolver)

    static tflite::MicroInterpreter static_interpreter(model, op_resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    // Allocate memory from the tensor_arena for the model's tensors.
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk){
        ESP_LOGE(TAG, "AllocateTensors() failed");
        return;
    }
    printModelInfo();
}

float input_scale = 0.007843103259801865;
int input_zero_pt = 0;

float output_scale = 0.03959125280380249;
int output_zero_pt = -3;

void loop() {
    digitalWrite(LED_BUILTIN, HIGH);
    delay(200);
    digitalWrite(LED_BUILTIN, LOW);
    delay(100);

    // //generate 2 random floats between 0 and 1
    // float x = random(-2048, 2048) / 2048.0;
    // float y = random(-2048, 2048) / 2048.0;

    //generate 42 random floats between 0 and 1
    float inputs[42];
    for (int i = 0; i < 42; i++){
        inputs[i] = random(-2048, 2048) / 2048.0;
    }

    TfLiteTensor *input = interpreter->input(0);
    for (int i = 0; i < 42; i++){
        input->data.int8[i] = int8_t(constrain(round(inputs[i] / input_scale + input_zero_pt), -128, 127));
    }

    long start = micros();
    if (interpreter->Invoke() == kTfLiteOk){
        Serial.printf("Invoke in: %d microseconds\n", micros() - start);
    }else{
        Serial.printf("Invoke failed!\n");
        return;
    }

    int8_t output_int = interpreter->output(0)->data.int8[0];
    float output_float = (output_int - output_zero_pt) * output_scale;

    Serial.print("Input: [");
    for (int i = 0; i < 42; i++){
        Serial.printf("%f, ", inputs[i]);
    }
    Serial.printf("]\n");

    Serial.printf("output_int: %d\n", output_int);
    Serial.printf("output_float: %f\n", output_float);
    Serial.print("\t\n");
}
