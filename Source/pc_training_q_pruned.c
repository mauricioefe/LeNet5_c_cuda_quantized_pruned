#include "lenet_q_pruned.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

//Update your paths!
#define FILE_TRAIN_IMAGE		"X:\\Development\\LeNet5_c_cuda_quantized_pruned\\Library\\LeNet5\\LeNet-5\\train-images-idx3-ubyte"
#define FILE_TRAIN_LABEL		"X:\\Development\\LeNet5_c_cuda_quantized_pruned\\Library\\LeNet5\\LeNet-5\\train-labels-idx1-ubyte"
#define FILE_TEST_IMAGE		"X:\\Development\\LeNet5_c_cuda_quantized_pruned\\Library\\LeNet5\\LeNet-5\\t10k-images-idx3-ubyte"
#define FILE_TEST_LABEL		"X:\\Development\\LeNet5_c_cuda_quantized_pruned\\Library\\LeNet5\\LeNet-5\\t10k-labels-idx1-ubyte"
#define LENET_FILE 		    "X:\\Development\\LeNet5_c_cuda_quantized_pruned\\Output\\model_pruned.dat"
#define LENET_Q_FILE 		"X:\\Development\\LeNet5_c_cuda_quantized_pruned\\Output\\model_q_pruned.dat"
#define COUNT_TRAIN		60000
#define COUNT_TEST		10000

int read_data(unsigned char(*data)[28][28], unsigned char label[], const int count, const char data_file[], const char label_file[])
{
    FILE *fp_image = fopen(data_file, "rb");
    FILE *fp_label = fopen(label_file, "rb");
    if (!fp_image||!fp_label) return 1;
	fseek(fp_image, 16, SEEK_SET);
	fseek(fp_label, 8, SEEK_SET);
	fread(data, sizeof(*data)*count, 1, fp_image);
	fread(label,count, 1, fp_label);
	fclose(fp_image);
	fclose(fp_label);
	return 0;
}

void training(LeNet5 *lenet,
              image *train_data,
              uint8 *train_label,
              int batch_size,
              int total_size,
              double prune_rate,
              int prune_frequency)
{
    int total_batches = total_size / batch_size;
    int warmup_batches = total_batches / 10;
    int percent = 0;

    for (int i = 0; i <= total_size - batch_size; i += batch_size) {
        int batch_idx = i / batch_size;
        TrainBatch(lenet,
                   train_data + i,
                   train_label + i,
                   batch_size);

        if (batch_idx * 100 / total_batches > percent) {
            percent = batch_idx * 100 / total_batches;
            printf("batchsize:%d\ttrain:%2d%%\n", batch_size, percent);
        }

        if (batch_idx >= warmup_batches &&
            ((batch_idx - warmup_batches) % prune_frequency) == 0)
        {
            printf("Pruning weights at batch %d (%.2f%%)\n",
                   batch_idx,
                   prune_rate * 100);
            prune_weights(lenet, prune_rate);
        }
    }
}

int testing(LeNet5 *lenet, image *test_data, uint8 *test_label, int total_size)
{
    int right = 0, percent = 0;
    for (int i = 0; i < total_size; ++i) {
        uint8 l = test_label[i];
        int p = Predict(lenet, test_data[i], 10);
        right += (l == p);
        if (i * 100 / total_size > percent)
            printf("test:%2d%%\n", percent = i * 100 / total_size);
    }
    return right;
}

int save(LeNet5 *lenet, const char filename[])
{
    FILE *fp = fopen(filename, "wb");
    if (!fp) return 1;
    fwrite(lenet, sizeof(LeNet5), 1, fp);
    fclose(fp);
    return 0;
}

int load(LeNet5 *lenet, const char filename[])
{
    FILE *fp = fopen(filename, "rb");
    if (!fp) return 1;
    fread(lenet, sizeof(LeNet5), 1, fp);
    fclose(fp);
    return 0;
}

void run_all()
{
    // ——— load MNIST data ———
    image *train_data = calloc(COUNT_TRAIN, sizeof(image));
    uint8 *train_label = calloc(COUNT_TRAIN, sizeof(uint8));
    image *test_data  = calloc(COUNT_TEST,  sizeof(image));
    uint8 *test_label = calloc(COUNT_TEST,  sizeof(uint8));
    if (read_data(train_data, train_label, COUNT_TRAIN, FILE_TRAIN_IMAGE, FILE_TRAIN_LABEL)
     || read_data(test_data,  test_label,  COUNT_TEST,  FILE_TEST_IMAGE,  FILE_TEST_LABEL))
    {
        fprintf(stderr, "ERROR: could not load MNIST files\n");
        return;
    }

    // ——— allocate & init (or load) your QAT model ———
    LeNet5 *lenet = malloc(sizeof(LeNet5));
    if (load(lenet, LENET_Q_FILE)) {
        Initial(lenet);
        printf("Initialized new network\n");
    } else {
        printf("Loaded existing FP model\n");
    }

    // ——— train with pruning ———
    int batches[] = { 300 };
    clock_t t0 = clock();
    for (size_t bi = 0; bi < sizeof(batches)/sizeof(batches[0]); ++bi) {
        training(lenet,
                 train_data,
                 train_label,
                 batches[bi],
                 COUNT_TRAIN,
                 PRUNE_RATE,
                 PRUNE_FREQUENCY);
    }
    printf("Training time: %.2f s\n",
           (clock() - t0)/(double)CLOCKS_PER_SEC);

    // ——— save models ———
    if (save(lenet, LENET_FILE) != 0) {
        fprintf(stderr, "Failed to save full-precision model\n");
    } else {
        printf("Full-precision model saved to %s\n", LENET_FILE);
    }

    if (save_quantized(lenet, LENET_Q_FILE) != 0) {
        fprintf(stderr, "ERROR: could not write quantized model to %s\n", LENET_Q_FILE);
    } else {
        printf("Quantized model written to %s\n", LENET_Q_FILE);
    }

    // ——— evaluate full-precision ———
    t0 = clock();
    int correct_fp = testing(lenet, test_data, test_label, COUNT_TEST);
    double infer_secs = (clock() - t0)/(double)CLOCKS_PER_SEC;
    printf("FP accuracy:        %d / %d (%.2f%%)\n",
           correct_fp, COUNT_TEST,
           correct_fp*100.0/COUNT_TEST);
    printf("FP inference: %.2f s total, %.2f ms/image\n",
           infer_secs, infer_secs*1000.0/COUNT_TEST);

    // ——— evaluate pure-int8 inference ———
    LeNet5Quant qnet;
    if (load_quantized(&qnet, LENET_Q_FILE) != 0) {
        fprintf(stderr, "ERROR: could not load quantized model\n");
    } else {
        int correct_q = 0;
        t0 = clock();
        for (int i = 0; i < COUNT_TEST; ++i) {
            uint8 pred = PredictQuant(&qnet, test_data[i]);
            correct_q += (pred == test_label[i]);
        }
        infer_secs = (clock() - t0)/(double)CLOCKS_PER_SEC;
        printf("QAT-int8 accuracy:  %d / %d (%.2f%%)\n",
               correct_q, COUNT_TEST,
               correct_q*100.0/COUNT_TEST);
        printf("QAT-int8 inference: %.2f s total, %.2f ms/image\n",
               infer_secs, infer_secs*1000.0/COUNT_TEST);
    }

    // ——— cleanup ———
    free(lenet);
    free(train_data);
    free(train_label);
    free(test_data);
    free(test_label);
}

int main()
{
    run_all();
    return 0;
}
