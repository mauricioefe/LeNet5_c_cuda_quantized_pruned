// lenet_quantized.c
#include "lenet_q_pruned.h"
#include <memory.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#define GETLENGTH(array) (sizeof(array)/sizeof(*(array)))
#define GETCOUNT(array)  (sizeof(array)/sizeof(double))
#define FOREACH(i,count) for (int i = 0; i < count; ++i)

#define CONVOLUTE_FULL(input,output,weight)												\
{																						\
	FOREACH(i0,GETLENGTH(input))														\
		FOREACH(i1,GETLENGTH(*(input)))													\
			FOREACH(w0,GETLENGTH(weight))												\
				FOREACH(w1,GETLENGTH(*(weight)))										\
					(output)[i0 + w0][i1 + w1] += (input)[i0][i1] * (weight)[w0][w1];	\
}

int8_t quantize(double value)
{
	if (value > QUANT_SCALE) return QUANT_SCALE;
	if (value < -QUANT_SCALE) return -QUANT_SCALE;
	return (int8_t)(value * QUANT_SCALE);
}

double dequantize(int8_t value)
{
	return (double)value / QUANT_SCALE;
}

#define CONVOLUTE_VALID(input, output, weight)                             \
{          																   \
    FOREACH(o0, GETLENGTH(output))                                         \
    FOREACH(o1, GETLENGTH(*(output)))                                      \
    FOREACH(w0, GETLENGTH(weight))                                         \
    FOREACH(w1, GETLENGTH(*(weight)))                                      \
        (output)[o0][o1] +=                                                \
            (input)[o0 + w0][o1 + w1] * qat_weight((weight)[w0][w1]);      \
}

#define CONVOLUTION_FORWARD(input, output, weight, bias, action)           \
{                                                                          \
    for (int in_ch = 0; in_ch < GETLENGTH(weight); ++in_ch)                \
        for (int out_ch = 0; out_ch < GETLENGTH(*(weight)); ++out_ch)      \
            CONVOLUTE_VALID(                                               \
                (input)[in_ch],                                            \
                (output)[out_ch],                                          \
                (weight)[in_ch][out_ch] );                                 \
                                                                           \
    FOREACH(j, GETLENGTH(bias))                                            \
        FOREACH(i, GETCOUNT(output[j]))                                    \
            ((double*)(output)[j])[i] =                                    \
                action(                                                    \
                  ((double*)(output)[j])[i] + qat_bias((bias)[j]) );       \
}

#define CONVOLUTION_BACKWARD(input,inerror,outerror,weight,wd,bd,actiongrad)\
{																			\
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
			CONVOLUTE_FULL(outerror[y], inerror[x], weight[x][y]);			\
	FOREACH(i, GETCOUNT(inerror))											\
		((double *)inerror)[i] *= actiongrad(((double *)input)[i]);			\
	FOREACH(j, GETLENGTH(outerror))											\
		FOREACH(i, GETCOUNT(outerror[j]))									\
		bd[j] += ((double *)outerror[j])[i];								\
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
			CONVOLUTE_VALID(input[x], wd[x][y], outerror[y]);				\
}


#define SUBSAMP_MAX_FORWARD(input,output)														\
{																								\
	const int len0 = GETLENGTH(*(input)) / GETLENGTH(*(output));								\
	const int len1 = GETLENGTH(**(input)) / GETLENGTH(**(output));								\
	FOREACH(i, GETLENGTH(output))																\
	FOREACH(o0, GETLENGTH(*(output)))															\
	FOREACH(o1, GETLENGTH(**(output)))															\
	{																							\
		int x0 = 0, x1 = 0, ismax;																\
		FOREACH(l0, len0)																		\
			FOREACH(l1, len1)																	\
		{																						\
			ismax = input[i][o0*len0 + l0][o1*len1 + l1] > input[i][o0*len0 + x0][o1*len1 + x1];\
			x0 += ismax * (l0 - x0);															\
			x1 += ismax * (l1 - x1);															\
		}																						\
		output[i][o0][o1] = input[i][o0*len0 + x0][o1*len1 + x1];								\
	}																							\
}

#define SUBSAMP_MAX_BACKWARD(input,inerror,outerror)											\
{																								\
	const int len0 = GETLENGTH(*(inerror)) / GETLENGTH(*(outerror));							\
	const int len1 = GETLENGTH(**(inerror)) / GETLENGTH(**(outerror));							\
	FOREACH(i, GETLENGTH(outerror))																\
	FOREACH(o0, GETLENGTH(*(outerror)))															\
	FOREACH(o1, GETLENGTH(**(outerror)))														\
	{																							\
		int x0 = 0, x1 = 0, ismax;																\
		FOREACH(l0, len0)																		\
			FOREACH(l1, len1)																	\
		{																						\
			ismax = input[i][o0*len0 + l0][o1*len1 + l1] > input[i][o0*len0 + x0][o1*len1 + x1];\
			x0 += ismax * (l0 - x0);															\
			x1 += ismax * (l1 - x1);															\
		}																						\
		inerror[i][o0*len0 + x0][o1*len1 + x1] = outerror[i][o0][o1];							\
	}																							\
}

#define DOT_PRODUCT_FORWARD(input, output, weight, bias, action)               \
{                                                                              \
    for (int x = 0; x < GETLENGTH(weight); ++x)                                \
        for (int j = 0; j < GETLENGTH(*(weight)); ++j)                         \
            ((double*)(output))[j] +=                                          \
                ((double*)(input))[x] * qat_weight( ((weight)[x][j]) );        \
    FOREACH(j, GETLENGTH(bias))                                                \
        ((double*)(output))[j] =                                               \
            action( ((double*)(output))[j] + qat_bias( (bias)[j] ) );          \
}

#define DOT_PRODUCT_BACKWARD(input,inerror,outerror,weight,wd,bd,actiongrad)	\
{																				\
	for (int x = 0; x < GETLENGTH(weight); ++x)									\
		for (int y = 0; y < GETLENGTH(*weight); ++y)							\
			((double *)inerror)[x] += ((double *)outerror)[y] * weight[x][y];	\
	FOREACH(i, GETCOUNT(inerror))												\
		((double *)inerror)[i] *= actiongrad(((double *)input)[i]);				\
	FOREACH(j, GETLENGTH(outerror))												\
		bd[j] += ((double *)outerror)[j];										\
	for (int x = 0; x < GETLENGTH(weight); ++x)									\
		for (int y = 0; y < GETLENGTH(*weight); ++y)							\
			wd[x][y] += ((double *)input)[x] * ((double *)outerror)[y];			\
}

double relu(double x)
{
	return x*(x > 0);
}

double relugrad(double y)
{
	return y > 0;
}

static int compare_doubles(const void *a, const void *b) {
    double da = *(const double*)a;
    double db = *(const double*)b;
    if (da < db) return -1;
    if (da > db) return +1;
    return 0;
}

// --------------------------------------------------------------------
// Pruning: collect all weights, sort by magnitude, pick percentile
// --------------------------------------------------------------------
void prune_weights(LeNet5 *lenet, double prune_rate) {
    // 1) count total weights
    size_t total_w =
        INPUT*LAYER1*LENGTH_KERNEL*LENGTH_KERNEL +
        LAYER2*LAYER3*LENGTH_KERNEL*LENGTH_KERNEL +
        LAYER4*LAYER5*LENGTH_KERNEL*LENGTH_KERNEL +
        (size_t)(LAYER5*LENGTH_FEATURE5*LENGTH_FEATURE5)*OUTPUT;

    // 2) gather abs(weights)
    double *buf = malloc(total_w * sizeof(double));
    size_t idx = 0;
    for (int i=0;i<INPUT;i++)
    for (int j=0;j<LAYER1;j++)
    for (int u=0;u<LENGTH_KERNEL;u++)
    for (int v=0;v<LENGTH_KERNEL;v++)
        buf[idx++] = fabs(lenet->weight0_1[i][j][u][v]);
    for (int i=0;i<LAYER2;i++)
    for (int j=0;j<LAYER3;j++)
    for (int u=0;u<LENGTH_KERNEL;u++)
    for (int v=0;v<LENGTH_KERNEL;v++)
        buf[idx++] = fabs(lenet->weight2_3[i][j][u][v]);
    for (int i=0;i<LAYER4;i++)
    for (int j=0;j<LAYER5;j++)
    for (int u=0;u<LENGTH_KERNEL;u++)
    for (int v=0;v<LENGTH_KERNEL;v++)
        buf[idx++] = fabs(lenet->weight4_5[i][j][u][v]);
    {
      size_t flat = LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5;
      for (size_t f=0; f<flat; f++)
          for (int j=0;j<OUTPUT;j++)
              buf[idx++] = fabs(lenet->weight5_6[f][j]);
    }

    // 3) find threshold
    qsort(buf, total_w, sizeof(double), compare_doubles);
    double threshold = buf[(size_t)(prune_rate * total_w)];
    free(buf);

    // 4) zero out small weights
    for (int i=0;i<INPUT;i++)
    for (int j=0;j<LAYER1;j++)
    for (int u=0;u<LENGTH_KERNEL;u++)
    for (int v=0;v<LENGTH_KERNEL;v++)
        if (fabs(lenet->weight0_1[i][j][u][v]) < threshold)
            lenet->weight0_1[i][j][u][v] = 0.0;
    for (int i=0;i<LAYER2;i++)
    for (int j=0;j<LAYER3;j++)
    for (int u=0;u<LENGTH_KERNEL;u++)
    for (int v=0;v<LENGTH_KERNEL;v++)
        if (fabs(lenet->weight2_3[i][j][u][v]) < threshold)
            lenet->weight2_3[i][j][u][v] = 0.0;
    for (int i=0;i<LAYER4;i++)
    for (int j=0;j<LAYER5;j++)
    for (int u=0;u<LENGTH_KERNEL;u++)
    for (int v=0;v<LENGTH_KERNEL;v++)
        if (fabs(lenet->weight4_5[i][j][u][v]) < threshold)
            lenet->weight4_5[i][j][u][v] = 0.0;
    {
      size_t flat = LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5;
      for (size_t f=0; f<flat; f++)
          for (int j=0;j<OUTPUT;j++)
              if (fabs(lenet->weight5_6[f][j]) < threshold)
                  lenet->weight5_6[f][j] = 0.0;
    }
}

static void forward(LeNet5 *lenet, Feature *features, double(*action)(double))
{
	CONVOLUTION_FORWARD(features->input, features->layer1, lenet->weight0_1, lenet->bias0_1, action);
	SUBSAMP_MAX_FORWARD(features->layer1, features->layer2);
	CONVOLUTION_FORWARD(features->layer2, features->layer3, lenet->weight2_3, lenet->bias2_3, action);
	SUBSAMP_MAX_FORWARD(features->layer3, features->layer4);
	CONVOLUTION_FORWARD(features->layer4, features->layer5, lenet->weight4_5, lenet->bias4_5, action);
	DOT_PRODUCT_FORWARD(features->layer5, features->output, lenet->weight5_6, lenet->bias5_6, action);
}

static void backward(LeNet5 *lenet, LeNet5 *deltas, Feature *errors, Feature *features, double(*actiongrad)(double))
{
	DOT_PRODUCT_BACKWARD(features->layer5, errors->layer5, errors->output, lenet->weight5_6, deltas->weight5_6, deltas->bias5_6, actiongrad);
	CONVOLUTION_BACKWARD(features->layer4, errors->layer4, errors->layer5, lenet->weight4_5, deltas->weight4_5, deltas->bias4_5, actiongrad);
	SUBSAMP_MAX_BACKWARD(features->layer3, errors->layer3, errors->layer4);
	CONVOLUTION_BACKWARD(features->layer2, errors->layer2, errors->layer3, lenet->weight2_3, deltas->weight2_3, deltas->bias2_3, actiongrad);
	SUBSAMP_MAX_BACKWARD(features->layer1, errors->layer1, errors->layer2);
	CONVOLUTION_BACKWARD(features->input, errors->input, errors->layer1, lenet->weight0_1, deltas->weight0_1, deltas->bias0_1, actiongrad);
}

static inline void load_input(Feature *features, image input)
{
	double (*layer0)[LENGTH_FEATURE0][LENGTH_FEATURE0] = features->input;
	const long sz = sizeof(image) / sizeof(**input);
	double mean = 0, std = 0;
	FOREACH(j, sizeof(image) / sizeof(*input))
		FOREACH(k, sizeof(*input) / sizeof(**input))
	{
		mean += input[j][k];
		std += input[j][k] * input[j][k];
	}
	mean /= sz;
	std = sqrt(std / sz - mean*mean);
	FOREACH(j, sizeof(image) / sizeof(*input))
		FOREACH(k, sizeof(*input) / sizeof(**input))
	{
		layer0[0][j + PADDING][k + PADDING] = (input[j][k] - mean) / std;
	}
}

static inline void softmax(double input[OUTPUT], double loss[OUTPUT], int label, int count)
{
	double inner = 0;
	for (int i = 0; i < count; ++i)
	{
		double res = 0;
		for (int j = 0; j < count; ++j)
		{
			res += exp(input[j] - input[i]);
		}
		loss[i] = 1. / res;
		inner -= loss[i] * loss[i];
	}
	inner += loss[label];
	for (int i = 0; i < count; ++i)
	{
		loss[i] *= (i == label) - loss[i] - inner;
	}
}

static void load_target(Feature *features, Feature *errors, int label)
{
	double *output = (double *)features->output;
	double *error = (double *)errors->output;
	softmax(output, error, label, GETCOUNT(features->output));
}

static uint8 get_result(Feature *features, uint8 count)
{
	double *output = (double *)features->output; 
	const int outlen = GETCOUNT(features->output);
	uint8 result = 0;
	double maxvalue = *output;
	for (uint8 i = 1; i < count; ++i)
	{
		if (output[i] > maxvalue)
		{
			maxvalue = output[i];
			result = i;
		}
	}
	return result;
}

static double f64rand()
{
	static int randbit = 0;
	if (!randbit)
	{
		srand((unsigned)time(0));
		for (int i = RAND_MAX; i; i >>= 1, ++randbit);
	}
	unsigned long long lvalue = 0x4000000000000000L;
	int i = 52 - randbit;
	for (; i > 0; i -= randbit)
		lvalue |= (unsigned long long)rand() << i;
	lvalue |= (unsigned long long)rand() >> -i;
	return *(double *)&lvalue - 3;
}


void TrainBatch(LeNet5 *lenet, image *inputs, uint8 *labels, int batchSize)
{
	double buffer[GETCOUNT(LeNet5)] = { 0 };
	int i = 0;
#pragma omp parallel for
	for (i = 0; i < batchSize; ++i)
	{
		Feature features = { 0 };
		Feature errors = { 0 };
		LeNet5	deltas = { 0 };
		load_input(&features, inputs[i]);
		forward(lenet, &features, relu);
		load_target(&features, &errors, labels[i]);
		backward(lenet, &deltas, &errors, &features, relugrad);
		#pragma omp critical
		{
			FOREACH(j, GETCOUNT(LeNet5))
				buffer[j] += ((double *)&deltas)[j];
		}
	}
	double k = ALPHA / batchSize;
	FOREACH(i, GETCOUNT(LeNet5))
		((double *)lenet)[i] += k * buffer[i];
}

void Train(LeNet5 *lenet, image input, uint8 label)
{
	Feature features = { 0 };
	Feature errors = { 0 };
	LeNet5 deltas = { 0 };
	load_input(&features, input);
	forward(lenet, &features, relu);
	load_target(&features, &errors, label);
	backward(lenet, &deltas, &errors, &features, relugrad);
	FOREACH(i, GETCOUNT(LeNet5))
		((double *)lenet)[i] += ALPHA * ((double *)&deltas)[i];
}

uint8 Predict(LeNet5 *lenet, image input,uint8 count)
{
	Feature features = { 0 };
	load_input(&features, input);
	forward(lenet, &features, relu);
	return get_result(&features, count);
}

void Initial(LeNet5 *lenet)
{
	for (double *pos = (double *)lenet->weight0_1; pos < (double *)lenet->bias0_1; *pos++ = f64rand());
	for (double *pos = (double *)lenet->weight0_1; pos < (double *)lenet->weight2_3; *pos++ *= sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (INPUT + LAYER1))));
	for (double *pos = (double *)lenet->weight2_3; pos < (double *)lenet->weight4_5; *pos++ *= sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER2 + LAYER3))));
	for (double *pos = (double *)lenet->weight4_5; pos < (double *)lenet->weight5_6; *pos++ *= sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER4 + LAYER5))));
	for (double *pos = (double *)lenet->weight5_6; pos < (double *)lenet->bias0_1; *pos++ *= sqrt(6.0 / (LAYER5 + OUTPUT)));
	for (int *pos = (int *)lenet->bias0_1; pos < (int *)(lenet + 1); *pos++ = 0);
}

int save_quantized(LeNet5 *lenet, const char *fname) {
    LeNet5Quant qnet;

    // 1) conv layer 0→1
    for (int i = 0; i < INPUT;  ++i)
    for (int j = 0; j < LAYER1; ++j)
    for (int u = 0; u < LENGTH_KERNEL; ++u)
    for (int v = 0; v < LENGTH_KERNEL; ++v)
        qnet.weight0_1[i][j][u][v] =
            quantize(lenet->weight0_1[i][j][u][v]);

    // 2) conv layer 2→3
    for (int i = 0; i < LAYER2; ++i)
    for (int j = 0; j < LAYER3; ++j)
    for (int u = 0; u < LENGTH_KERNEL; ++u)
    for (int v = 0; v < LENGTH_KERNEL; ++v)
        qnet.weight2_3[i][j][u][v] =
            quantize(lenet->weight2_3[i][j][u][v]);

    // 3) conv layer 4→5
    for (int i = 0; i < LAYER4; ++i)
    for (int j = 0; j < LAYER5; ++j)
    for (int u = 0; u < LENGTH_KERNEL; ++u)
    for (int v = 0; v < LENGTH_KERNEL; ++v)
        qnet.weight4_5[i][j][u][v] =
            quantize(lenet->weight4_5[i][j][u][v]);

    // 4) fully-connected layer 5→6
    {
      int flat = LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5;
      for (int idx = 0; idx < flat; ++idx)
      for (int j   = 0; j < OUTPUT;            ++j)
          qnet.weight5_6[idx][j] =
              quantize(lenet->weight5_6[idx][j]);
    }

    // 5) biases
    for (int j = 0; j < LAYER1; ++j) qnet.bias0_1[j] = quantize(lenet->bias0_1[j]);
    for (int j = 0; j < LAYER3; ++j) qnet.bias2_3[j] = quantize(lenet->bias2_3[j]);
    for (int j = 0; j < LAYER5; ++j) qnet.bias4_5[j] = quantize(lenet->bias4_5[j]);
    for (int j = 0; j < OUTPUT;  ++j) qnet.bias5_6[j] = quantize(lenet->bias5_6[j]);

    // 6) write to file
    FILE *fp = fopen(fname, "wb");
    if (!fp) return 1;
    fwrite(&qnet, sizeof(LeNet5Quant), 1, fp);
    fclose(fp);
    return 0;
}

int load_quantized(LeNet5Quant *qnet, const char *fname) {
    FILE *fp = fopen(fname, "rb");
    if (!fp) return 1;
    fread(qnet, sizeof(LeNet5Quant), 1, fp);
    fclose(fp);
    return 0;
}

uint8 PredictQuant(const LeNet5Quant *qnet, image input) {
    Feature F = {0};

    // -- normalize & load image into F.input[0][..][..]
    load_input(&F, input);

    // -- conv1 → relu
    for (int out_c = 0; out_c < LAYER1; ++out_c) {
      for (int i0 = 0; i0 < LENGTH_FEATURE1; ++i0) {
      for (int i1 = 0; i1 < LENGTH_FEATURE1; ++i1) {
        double acc = 0.0;
        for (int in_c = 0; in_c < INPUT; ++in_c)
        for (int w0 = 0;   w0 < LENGTH_KERNEL; ++w0)
        for (int w1 = 0;   w1 < LENGTH_KERNEL; ++w1)
          acc += F.input[in_c][i0 + w0][i1 + w1]
               * dequantize(qnet->weight0_1[in_c][out_c][w0][w1]);
        F.layer1[out_c][i0][i1] = relu(acc
                                  + dequantize(qnet->bias0_1[out_c]));
      }}
    }

    // -- pool1
    SUBSAMP_MAX_FORWARD(F.layer1, F.layer2);

    // -- conv2 → relu
    for (int out_c = 0; out_c < LAYER3; ++out_c) {
      for (int i0 = 0; i0 < LENGTH_FEATURE3; ++i0) {
      for (int i1 = 0; i1 < LENGTH_FEATURE3; ++i1) {
        double acc = 0.0;
        for (int in_c = 0; in_c < LAYER2; ++in_c)
        for (int w0 = 0;   w0 < LENGTH_KERNEL; ++w0)
        for (int w1 = 0;   w1 < LENGTH_KERNEL; ++w1)
          acc += F.layer2[in_c][i0 + w0][i1 + w1]
               * dequantize(qnet->weight2_3[in_c][out_c][w0][w1]);
        F.layer3[out_c][i0][i1] = relu(acc
                                  + dequantize(qnet->bias2_3[out_c]));
      }}
    }

    // -- pool2
    SUBSAMP_MAX_FORWARD(F.layer3, F.layer4);

    // -- conv3 → relu
    for (int out_c = 0; out_c < LAYER5; ++out_c) {
      for (int i0 = 0; i0 < LENGTH_FEATURE5; ++i0) {
      for (int i1 = 0; i1 < LENGTH_FEATURE5; ++i1) {
        double acc = 0.0;
        for (int in_c = 0; in_c < LAYER4; ++in_c)
        for (int w0 = 0;   w0 < LENGTH_KERNEL; ++w0)
        for (int w1 = 0;   w1 < LENGTH_KERNEL; ++w1)
          acc += F.layer4[in_c][i0 + w0][i1 + w1]
               * dequantize(qnet->weight4_5[in_c][out_c][w0][w1]);
        F.layer5[out_c][i0][i1] = relu(acc
                                  + dequantize(qnet->bias4_5[out_c]));
      }}
    }

    // -- fully-connected 5→6
    {
      int flat = LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5;
      double out_buf[OUTPUT] = {0};
      for (int idx = 0; idx < flat; ++idx) {
        int in_c = idx / (LENGTH_FEATURE5 * LENGTH_FEATURE5);
        int rem  = idx % (LENGTH_FEATURE5 * LENGTH_FEATURE5);
        int i0   = rem / LENGTH_FEATURE5;
        int i1   = rem % LENGTH_FEATURE5;
        double in_val = F.layer5[in_c][i0][i1];
        for (int j = 0; j < OUTPUT; ++j) {
          out_buf[j] += in_val
                      * dequantize(qnet->weight5_6[idx][j]);
        }
      }
      for (int j = 0; j < OUTPUT; ++j) {
        F.output[j] = relu(out_buf[j]
                       + dequantize(qnet->bias5_6[j]));
      }
    }

    // -- pick the max response
    return get_result(&F, OUTPUT);
}
