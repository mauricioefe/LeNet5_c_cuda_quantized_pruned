import numpy as np

LENET_FILE 	    =	"X:\\Development\\LeNet5_c_cuda_quantized_pruned\\Output\\model_pruned.dat"
LENET_Q_FILE 	=	"X:\\Development\\LeNet5_c_cuda_quantized_pruned\\Output\\model_q_pruned.dat"

for fname in (LENET_FILE, LENET_Q_FILE):
    data = np.fromfile(fname, dtype=np.int8)
    sparsity = 100.0 * np.mean(data == 0)
    print(f"{fname}: {sparsity:.2f}% zeros")