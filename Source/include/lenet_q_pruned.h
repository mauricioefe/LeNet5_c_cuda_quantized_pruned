// lenet_quantized.h
#pragma once

#include <stdint.h>
#include <stdlib.h>
#include <math.h>

#define LENGTH_KERNEL   5
#define LENGTH_FEATURE0	32
#define LENGTH_FEATURE1	(LENGTH_FEATURE0 - LENGTH_KERNEL + 1)
#define LENGTH_FEATURE2	(LENGTH_FEATURE1 >> 1)
#define LENGTH_FEATURE3	(LENGTH_FEATURE2 - LENGTH_KERNEL + 1)
#define	LENGTH_FEATURE4	(LENGTH_FEATURE3 >> 1)
#define LENGTH_FEATURE5	(LENGTH_FEATURE4 - LENGTH_KERNEL + 1)

#define INPUT			1
#define LAYER1			6
#define LAYER2			6
#define LAYER3			16
#define LAYER4			16
#define LAYER5			120
#define OUTPUT          10

#define ALPHA           0.5
#define PADDING         2
#define QUANT_SCALE     127.0

#define PRUNE_RATE        0.2    // 20% of the time
#define PRUNE_FREQUENCY   100    // every 100 batches

typedef unsigned char uint8;
typedef uint8 image[28][28];

typedef struct LeNet5 {
    double weight0_1[INPUT][LAYER1][LENGTH_KERNEL][LENGTH_KERNEL];
    double weight2_3[LAYER2][LAYER3][LENGTH_KERNEL][LENGTH_KERNEL];
    double weight4_5[LAYER4][LAYER5][LENGTH_KERNEL][LENGTH_KERNEL];
    double weight5_6[LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5][OUTPUT];

    double bias0_1[LAYER1];
    double bias2_3[LAYER3];
    double bias4_5[LAYER5];
    double bias5_6[OUTPUT];
} LeNet5;

typedef struct LeNet5Quant {
    int8_t weight0_1[INPUT][LAYER1][LENGTH_KERNEL][LENGTH_KERNEL];
    int8_t weight2_3[LAYER2][LAYER3][LENGTH_KERNEL][LENGTH_KERNEL];
    int8_t weight4_5[LAYER4][LAYER5][LENGTH_KERNEL][LENGTH_KERNEL];
    int8_t weight5_6[LAYER5*LENGTH_FEATURE5*LENGTH_FEATURE5][OUTPUT];

    int8_t bias0_1[LAYER1];
    int8_t bias2_3[LAYER3];
    int8_t bias4_5[LAYER5];
    int8_t bias5_6[OUTPUT];
} LeNet5Quant;

typedef struct Feature {
    double input[INPUT][LENGTH_FEATURE0][LENGTH_FEATURE0];
    double layer1[LAYER1][LENGTH_FEATURE1][LENGTH_FEATURE1];
    double layer2[LAYER2][LENGTH_FEATURE2][LENGTH_FEATURE2];
    double layer3[LAYER3][LENGTH_FEATURE3][LENGTH_FEATURE3];
    double layer4[LAYER4][LENGTH_FEATURE4][LENGTH_FEATURE4];
    double layer5[LAYER5][LENGTH_FEATURE5][LENGTH_FEATURE5];
    double output[OUTPUT];
} Feature;

int read_data(unsigned char(*data)[28][28], unsigned char label[], int, const char[], const char[]);
void TrainBatch(LeNet5 *lenet, image *inputs, uint8 *labels, int batchSize);
uint8 Predict(LeNet5 *lenet, image input, uint8 count);
void Initial(LeNet5 *lenet);
int save_quantized(LeNet5 *lenet, const char *fname);
int load_quantized(LeNet5Quant *qnet, const char *fname);
uint8 PredictQuant(const LeNet5Quant *qnet, image input);

int8_t quantize(double value);
double dequantize(int8_t value);
static inline double qat_weight(double w) { return dequantize(quantize(w)); }
static inline double qat_bias  (double b) { return dequantize(quantize(b)); }

void prune_weights(LeNet5 *lenet, double prune_rate);
