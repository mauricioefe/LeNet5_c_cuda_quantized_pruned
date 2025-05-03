#include "lenet_q_pruned.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// ——— file paths & constants ———
#define FILE_TEST_IMAGE   "t10k-images-idx3-ubyte"
#define FILE_TEST_LABEL   "t10k-labels-idx1-ubyte"
#define LENET_FP_FILE     "model_pruned.dat"
#define LENET_Q_FILE      "model_q_pruned.dat"
#define COUNT_TEST        10000

// ——— helpers ———
int read_data(image data[], uint8 label[], int count,
              const char *data_file, const char *label_file)
{
    FILE *fimg = fopen(data_file, "rb");
    FILE *flab = fopen(label_file, "rb");
    if (!fimg || !flab) return 1;
    fseek(fimg, 16, SEEK_SET);
    fseek(flab, 8,  SEEK_SET);
    fread(data,  sizeof(image)*count, 1, fimg);
    fread(label, sizeof(uint8)*count,  1, flab);
    fclose(fimg);
    fclose(flab);
    return 0;
}

int load_fp_model(LeNet5 *lenet, const char *fname)
{
    FILE *fp = fopen(fname, "rb");
    if (!fp) return 1;
    fread(lenet, sizeof(LeNet5), 1, fp);
    fclose(fp);
    return 0;
}

// main entry
int main(void)
{
    // 1) load test set
    image *test_data  = malloc(sizeof(image)*COUNT_TEST);
    uint8 *test_label = malloc(sizeof(uint8)*COUNT_TEST);
    if (!test_data || !test_label ||
        read_data(test_data, test_label, COUNT_TEST,
                  FILE_TEST_IMAGE, FILE_TEST_LABEL))
    {
        fprintf(stderr, "ERROR: could not load MNIST test data\n");
        return 1;
    }

    // 2) full-precision inference
    LeNet5 *lenet = malloc(sizeof(LeNet5));
    if (!lenet || load_fp_model(lenet, LENET_FP_FILE)) {
        fprintf(stderr, "ERROR: could not load FP model (%s)\n", LENET_FP_FILE);
        return 1;
    }

    clock_t t0 = clock();
    int correct_fp = 0;
    for (int i = 0; i < COUNT_TEST; ++i) {
        uint8 pred = Predict(lenet, test_data[i], OUTPUT);
        if (pred == test_label[i]) ++correct_fp;
    }
    double fp_time = (clock() - t0) / (double)CLOCKS_PER_SEC;

    printf("FP  Accuracy: %d / %d (%.2f%%)\n",
           correct_fp, COUNT_TEST,
           correct_fp * 100.0 / COUNT_TEST);
    printf("FP  Time:     %.2f s total, %.2f ms/image\n\n",
           fp_time, fp_time*1000.0 / COUNT_TEST);

    // 3) quantized inference
    LeNet5Quant qnet;
    if (load_quantized(&qnet, LENET_Q_FILE)) {
        fprintf(stderr, "ERROR: could not load QAT model (%s)\n", LENET_Q_FILE);
        return 1;
    }

    t0 = clock();
    int correct_q = 0;
    for (int i = 0; i < COUNT_TEST; ++i) {
        uint8 pred = PredictQuant(&qnet, test_data[i]);
        if (pred == test_label[i]) ++correct_q;
    }
    double q_time = (clock() - t0) / (double)CLOCKS_PER_SEC;

    printf("QAT Accuracy: %d / %d (%.2f%%)\n",
           correct_q, COUNT_TEST,
           correct_q * 100.0 / COUNT_TEST);
    printf("QAT Time:     %.2f s total, %.2f ms/image\n",
           q_time, q_time*1000.0 / COUNT_TEST);

    // 4) cleanup
    free(lenet);
    free(test_data);
    free(test_label);
    return 0;
}