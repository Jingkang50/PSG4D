#include "otsu.h"
#include "../cuda_utils.h"

__global__ void get_hist_cuda_(int* input, int* hist, int step, int pts_num){
    const Int inst = blockIdx.x; 
    for(Int idx = threadIdx.x; idx<pts_num ; idx += blockDim.x){
        int thre = input[inst * pts_num + idx];
        hist[blockIdx.x * step + thre] += 1;
    }
}


void get_hist_cuda(int* input, int* hist, int step, int inst_num, int pts_num){
    cudaError_t err;

    dim3 blocks(inst_num);
    dim3 threads(THREADS_PER_BLOCK);

    int* p_hist;

    cudaMalloc((void**)&p_hist, inst_num*step*sizeof(Int));
    cudaMemcpy(p_hist, hist, inst_num*step*sizeof(Int), cudaMemcpyHostToDevice);

    get_hist_cuda_<<<blocks, threads>>>(input, p_hist, step, pts_num);

    cudaMemcpy(hist, p_hist, inst_num*step*sizeof(Int), cudaMemcpyDeviceToHost);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}