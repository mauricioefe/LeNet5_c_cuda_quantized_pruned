#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

// ——— External files ————————————————————————————————————————————————
#define FILE_TEST_IMAGE   "X:\\Development\\LeNet5_c_cuda_quantized_pruned\\Library\\LeNet5\\LeNet-5\\t10k-images-idx3-ubyte"
#define FILE_TEST_LABEL   "X:\\Development\\LeNet5_c_cuda_quantized_pruned\\Library\\LeNet5\\LeNet-5\\t10k-labels-idx1-ubyte"
#define LENET_Q_FILE      "X:\\Development\\LeNet5_c_cuda_quantized_pruned\\Output\\model_quantized.dat"

// ——— Constants & Shapes —————————————————————————————————————————————
#define COUNT_TEST        10000
#define LENGTH_KERNEL     5
#define PADDING           2
#define LENGTH_FEATURE0   (28 + 2*PADDING)
#define LENGTH_FEATURE1   (LENGTH_FEATURE0 - LENGTH_KERNEL + 1)
#define LENGTH_FEATURE2   (LENGTH_FEATURE1 >> 1)
#define LENGTH_FEATURE3   (LENGTH_FEATURE2 - LENGTH_KERNEL + 1)
#define LENGTH_FEATURE4   (LENGTH_FEATURE3 >> 1)
#define LENGTH_FEATURE5   (LENGTH_FEATURE4 - LENGTH_KERNEL + 1)

#define INPUT             1
#define LAYER1            6
#define LAYER2            6
#define LAYER3            16
#define LAYER4            16
#define LAYER5            120
#define OUTPUT            10

#define QUANT_SCALE       127.0

typedef unsigned char uint8;
typedef uint8         image[28][28];

// ——— Quantized model struct ————————————————————————————————————————
typedef struct {
    int8_t weight0_1[INPUT][LAYER1][LENGTH_KERNEL][LENGTH_KERNEL];
    int8_t weight2_3[LAYER2][LAYER3][LENGTH_KERNEL][LENGTH_KERNEL];
    int8_t weight4_5[LAYER4][LAYER5][LENGTH_KERNEL][LENGTH_KERNEL];
    int8_t weight5_6[LAYER5*LENGTH_FEATURE5*LENGTH_FEATURE5][OUTPUT];
    int8_t bias0_1[LAYER1];
    int8_t bias2_3[LAYER3];
    int8_t bias4_5[LAYER5];
    int8_t bias5_6[OUTPUT];
} LeNet5Quant;

// ——— Intermediate feature maps struct —————————————————————————————
typedef struct {
    double input [INPUT][LENGTH_FEATURE0][LENGTH_FEATURE0];
    double layer1[LAYER1][LENGTH_FEATURE1][LENGTH_FEATURE1];
    double layer2[LAYER2][LENGTH_FEATURE2][LENGTH_FEATURE2];
    double layer3[LAYER3][LENGTH_FEATURE3][LENGTH_FEATURE3];
    double layer4[LAYER4][LENGTH_FEATURE4][LENGTH_FEATURE4];
    double layer5[LAYER5][LENGTH_FEATURE5][LENGTH_FEATURE5];
    double output[OUTPUT];
} Feature;

// ——— Host helpers ———————————————————————————————————————————————
int read_data(image data[], uint8 label[], int count,
              const char *data_file, const char *label_file)
{
    FILE *fimg = fopen(data_file, "rb"), *flab = fopen(label_file, "rb");
    if (!fimg || !flab) return 1;
    fseek(fimg, 16, SEEK_SET);
    fseek(flab,  8,  SEEK_SET);
    fread(data,  sizeof(image)*count,  1, fimg);
    fread(label, sizeof(uint8)*count,  1, flab);
    fclose(fimg); fclose(flab);
    return 0;
}

int load_quantized(LeNet5Quant *qnet, const char *fname)
{
    FILE *f = fopen(fname, "rb");
    if (!f) return 1;
    fread(qnet, sizeof(LeNet5Quant), 1, f);
    fclose(f);
    return 0;
}

// ——— Device utilities ————————————————————————————————————————————
__device__ inline double dev_dequant(int8_t v){ return (double)v/QUANT_SCALE; }
__device__ inline double dev_relu(double x){ return x>0 ? x : 0; }

// valid convolution + dequant
__device__ void conv2d_valid_q(
    const double in[], int in_dim,
    double out[], int out_dim,
    const int8_t kernel[][LENGTH_KERNEL],
    int kernel_dim)
{
    // 2D→flattened index loops
    for (int oy = 0; oy < out_dim; ++oy) {
      for (int ox = 0; ox < out_dim; ++ox) {
        double acc = 0.0;
        for (int ky = 0; ky < kernel_dim; ++ky) {
          for (int kx = 0; kx < kernel_dim; ++kx) {
            double iv = in[(oy+ky)*in_dim + (ox+kx)];
            acc += iv * dev_dequant(kernel[ky][kx]);
          }
        }
        out[oy*out_dim + ox] += acc;
      }
    }
}

// max-pool 2×2
__device__ void maxpool2x2_q(
    const double in[], int in_dim,
    double out[], int out_dim)
{
    for (int oy = 0; oy < out_dim; ++oy) {
      for (int ox = 0; ox < out_dim; ++ox) {
        double m = in[(oy*2)*in_dim + (ox*2)];
        for (int dy = 0; dy < 2; ++dy)
        for (int dx = 0; dx < 2; ++dx)
          m = fmax(m, in[(oy*2+dy)*in_dim + (ox*2+dx)]);
        out[oy*out_dim + ox] = m;
      }
    }
}

// fully-connected
__device__ void fc_q(
    const double in[], int in_size,
    double out[], int out_size,
    const int8_t weight[][OUTPUT],
    const int8_t bias[])
{
    for (int j = 0; j < out_size; ++j) {
      double acc = dev_dequant(bias[j]);
      for (int i = 0; i < in_size; ++i) {
        acc += in[i] * dev_dequant(weight[i][j]);
      }
      out[j] = dev_relu(acc);
    }
}

// single-image forward
__device__ uint8 forward_cuda_q(const image img, const LeNet5Quant *qnet)
{
    // 1) normalize + pad into 32×32
    double pad[32][32] = {0}, mean=0, sq=0;
    for(int y=0;y<28;++y) for(int x=0;x<28;++x){
      mean += img[y][x]; sq += img[y][x]*img[y][x];
    }
    mean /= 784.0; double std = sqrt(sq/784.0 - mean*mean);
    for(int y=0;y<28;++y) for(int x=0;x<28;++x){
      pad[y+PADDING][x+PADDING] = (img[y][x]-mean)/std;
    }

    Feature f = {0};
    // load input
    for(int y=0;y<32;++y) for(int x=0;x<32;++x)
      f.input[0][y][x] = pad[y][x];

    // conv1→relu→pool (32→28→14)
    for(int oc=0;oc<LAYER1;++oc){
      // accumulate over INPUT=1 channel
      conv2d_valid_q(&f.input[0][0][0], 32,
                     &f.layer1[oc][0][0], 28,
                     qnet->weight0_1[0][oc], 5);
      // bias+relu
      for(int i=0;i<28*28;++i)
        f.layer1[oc][0][i] = dev_relu(f.layer1[oc][0][i] + dev_dequant(qnet->bias0_1[oc]));
      maxpool2x2_q(&f.layer1[oc][0][0], 28,
                   &f.layer2[oc][0][0], 14);
    }

    // conv2→relu→pool (14→10→5)
    for(int oc=0;oc<LAYER3;++oc){
      for(int ic=0;ic<LAYER2;++ic)
        conv2d_valid_q(&f.layer2[ic][0][0], 14,
                       &f.layer3[oc][0][0], 10,
                       qnet->weight2_3[ic][oc], 5);
      for(int i=0;i<10*10;++i)
        f.layer3[oc][0][i] = dev_relu(f.layer3[oc][0][i] + dev_dequant(qnet->bias2_3[oc]));
      maxpool2x2_q(&f.layer3[oc][0][0], 10,
                   &f.layer4[oc][0][0], 5);
    }

    // conv3→relu (5→1)
    for(int oc=0;oc<LAYER5;++oc){
      for(int ic=0;ic<LAYER4;++ic)
        conv2d_valid_q(&f.layer4[ic][0][0], 5,
                       &f.layer5[oc][0][0], 1,
                       qnet->weight4_5[ic][oc], 5);
      f.layer5[oc][0][0] = dev_relu(f.layer5[oc][0][0] + dev_dequant(qnet->bias4_5[oc]));
    }

    // flatten + fc
    double flat[LAYER5];
    for(int i=0;i<LAYER5;++i) flat[i] = f.layer5[i][0][0];
    fc_q(flat, LAYER5, f.output, OUTPUT,
         qnet->weight5_6, qnet->bias5_6);

    // argmax
    uint8 best = 0; double mv = f.output[0];
    for(int i=1;i<OUTPUT;++i){
      if(f.output[i]>mv){ mv=f.output[i]; best=i; }
    }
    return best;
}

// kernel: one thread per image
__global__ void predict_kernel_q(const image *imgs,
                                 const LeNet5Quant *qnet,
                                 uint8 *out)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx < COUNT_TEST)
        out[idx] = forward_cuda_q(imgs[idx], qnet);
}

// ——— Main ———————————————————————————————————————————————————————
int main()
{
    // 1) Host load quantized model
    LeNet5Quant *h_qnet = (LeNet5Quant*)malloc(sizeof(LeNet5Quant));
    if(!h_qnet || load_quantized(h_qnet, LENET_Q_FILE)){
        fprintf(stderr, "ERROR loading %s\n", LENET_Q_FILE);
        return 1;
    }

    // 2) Host load MNIST test
    image  *h_imgs   = (image*)malloc(sizeof(image)*COUNT_TEST);
    uint8  *h_labels = (uint8*)malloc(sizeof(uint8)*COUNT_TEST);
    if(!h_imgs || !h_labels ||
       read_data(h_imgs, h_labels, COUNT_TEST, FILE_TEST_IMAGE, FILE_TEST_LABEL))
    {
        fprintf(stderr, "ERROR loading test data\n");
        return 1;
    }

    // 3) Device alloc & copy
    image        *d_imgs;  cudaMalloc(&d_imgs, sizeof(image)*COUNT_TEST);
    uint8        *d_out;   cudaMalloc(&d_out,  sizeof(uint8)*COUNT_TEST);
    LeNet5Quant  *d_qnet;  cudaMalloc(&d_qnet, sizeof(LeNet5Quant));

    cudaMemcpy(d_imgs,  h_imgs,  sizeof(image)*COUNT_TEST, cudaMemcpyHostToDevice);
    cudaMemcpy(d_qnet,  h_qnet,  sizeof(LeNet5Quant),    cudaMemcpyHostToDevice);

    // 4) Launch kernel + time
    clock_t t0 = clock();
    predict_kernel_q<<<(COUNT_TEST+255)/256, 256>>>(d_imgs, d_qnet, d_out);
    cudaDeviceSynchronize();
    double elapsed = (clock() - t0)/(double)CLOCKS_PER_SEC;

    // 5) Copy back & accuracy
    uint8 *h_out = (uint8*)malloc(sizeof(uint8)*COUNT_TEST);
    cudaMemcpy(h_out, d_out, sizeof(uint8)*COUNT_TEST, cudaMemcpyDeviceToHost);

    int correct = 0;
    for(int i=0; i<COUNT_TEST; ++i)
        if(h_out[i] == h_labels[i]) ++correct;

    printf("QAT-int8 CUDA Accuracy: %d / %d (%.2f%%)\n",
           correct, COUNT_TEST, correct*100.0/COUNT_TEST);
    printf("Elapsed Time: %.2f seconds\n", elapsed);

    // 6) Cleanup
    cudaFree(d_imgs);
    cudaFree(d_out);
    cudaFree(d_qnet);
    free(h_qnet);
    free(h_imgs);
    free(h_labels);
    free(h_out);

    return 0;
}

