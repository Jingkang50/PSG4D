/*
Ball Query with BatchIdx
Written by Li Jiang
All Rights Reserved 2020.
*/
#include "topk_nms.h"
#include "../cuda_utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void describe_candidates_cuda_(Int idx_o, float* heatmap, const float* coords, 
    const int* semantic_label, const float radius, int start, int end, int* visited,
    Int *idxs_f, int maxActive_f, Int *idxs_b, int maxActive_b, int *size_f, int *size_b) {
    float radius2 = radius * radius;
    float o_x = coords[idx_o * 3 + 0];
    float o_y = coords[idx_o * 3 + 1];
    float o_z = coords[idx_o * 3 + 2];
    int o_sem = semantic_label[idx_o];
    int pt_f = 0;
    int pt_b = 0;

    const Int idx = blockIdx.x * blockDim.x + threadIdx.x + start;
    if (idx < end)
    {   
        float x = coords[idx * 3 + 0];
        float y = coords[idx * 3 + 1];
        float z = coords[idx * 3 + 2];
        float d2 = (o_x - x) * (o_x - x) + (o_y - y) * (o_y - y) + (o_z - z) * (o_z - z);

        if (d2 < radius2){
            visited[idx - start] = 1;
            if ((o_sem == semantic_label[idx])){
                pt_f = atomicAdd(size_f, 1);
                if ((pt_f < maxActive_f)){  
                    //idxs_f[pt_f * 2 + 0] = idx_o;
                    idxs_f[pt_f] = idx;
                }
                //atomicMax(max_conf, (int)1e6*heatmap[idx]);
            }
        }
        if ((d2 < 4 * radius2) && (o_sem != semantic_label[idx])){
            pt_b = atomicAdd(size_b, 1);
            if ((pt_b < maxActive_b)){
                //idxs_b[pt_b * 2 + 0] = idx_o;
                idxs_b[pt_b] = idx;
            }
        } 
    }
}
 
int describe_candidates_cuda(Int idx,  float* heatmap, const float* coords, const int* semantic_label, int nPoint,
                             float local_thres, int start, int end, const float radius, int* visited,
                             Int *idxs_f, int *maxActive_f, Int *idxs_b, int *maxActive_b, cudaStream_t stream){
    // param xyz: (n, 3)
    // param batch_idxs: (n)
    // param batch_offsets: (B + 1)
    // output idx: (n * meanActive) dim 0 for number of points in the ball, idx in n
    // output start_len: (n, 2), int

    cudaError_t err;

    dim3 blocks(DIVUP(end-start, THREADS_PER_BLOCK));
    dim3 threads(THREADS_PER_BLOCK);

    Int nPoint_batch = end - start ;
    int size_f = 0;
    int size_b = 0;
    //float max_conf = heatmap[idx];

    int* p_size_f;
    int* p_size_b;
    Int* p_idxs_f;
    Int* p_idxs_b;
    //int* p_max_conf;
    int* p_visited;
    //float* p_heatmap;

    cudaMalloc((void**)&p_size_f, sizeof(int));
    cudaMalloc((void**)&p_size_b, sizeof(int));
    cudaMalloc((void**)&p_idxs_f, *maxActive_f*sizeof(Int));
    cudaMalloc((void**)&p_idxs_b, *maxActive_b*sizeof(Int));
    //cudaMalloc((void**)&p_max_conf, sizeof(int));
    //cudaMalloc((void**)&p_heatmap, nPoint*sizeof(float));
    cudaMalloc((void**)&p_visited, nPoint_batch*sizeof(int));

    cudaMemcpy(p_size_f, &size_f, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(p_size_b, &size_b, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(p_idxs_f, idxs_f, *maxActive_f*sizeof(Int), cudaMemcpyHostToDevice);
    cudaMemcpy(p_idxs_b, idxs_b, *maxActive_b*sizeof(Int), cudaMemcpyHostToDevice);
    //cudaMemcpy(p_max_conf, &max_conf, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(p_visited, visited, nPoint_batch*sizeof(int), cudaMemcpyHostToDevice);
    //cudaMemcpy(p_heatmap, heatmap, nPoint*sizeof(float), cudaMemcpyHostToDevice);

    //printf("nPoint: %d", nPoint);

    describe_candidates_cuda_<<<blocks, threads, 0, stream>>>(idx, heatmap, coords, semantic_label,
                            radius, start, end, p_visited, p_idxs_f, *maxActive_f, p_idxs_b, *maxActive_b, 
                            p_size_f, p_size_b);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    cudaMemcpy(&size_f, p_size_f, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&size_b, p_size_b, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(idxs_f, p_idxs_f, *maxActive_f*sizeof(Int), cudaMemcpyDeviceToHost);
    cudaMemcpy(idxs_b, p_idxs_b, *maxActive_b*sizeof(Int), cudaMemcpyDeviceToHost);
    cudaMemcpy(visited, p_visited, nPoint_batch*sizeof(int), cudaMemcpyDeviceToHost);
    //cudaMemcpy(&max_conf, p_max_conf, sizeof(int), cudaMemcpyDeviceToHost);
    //cudaMemcpy(heatmap, p_heatmap, nPoint*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(p_size_f);
    cudaFree(p_size_b);
    cudaFree(p_idxs_f);
    cudaFree(p_idxs_b);
    //cudaFree(p_max_conf);
    //cudaFree(p_heatmap);
    cudaFree(p_visited);
    
    //printf("center: %f, max: %f, size_f: %d, size_b: %d", heatmap[idx], (float(max_conf) / 10000), size_f, size_b);
    if (size_f > *maxActive_f) size_f = *maxActive_f;
    if (size_b > *maxActive_b) size_b = *maxActive_b;
    
    //printf("center: %f, max: %f, score: %f", (heatmap[idx]*1e6), (float)max_conf, (1e6*heatmap[idx] / (float)max_conf));

    *maxActive_f = size_f;
    *maxActive_b = size_b;

    return size_f;
}