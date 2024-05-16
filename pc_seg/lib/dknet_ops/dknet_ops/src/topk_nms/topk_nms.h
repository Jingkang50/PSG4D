/*
Ball Query with BatchIdx & Clustering Algorithm
Written by Li Jiang
All Rights Reserved 2020.
*/

#ifndef TOPK_NMS_H
#define TOPK_NMS_H
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THC.h>

#include "../datatype/datatype.h"

int describe_candidates_cuda(Int idx, float* heatmap, const float* coords, const int* semantic_label, int nPoint, float local_thres, int start, int end, const float radius, int* visited, Int *idxs_f, int *maxActive_f, Int *idxs_b, int *maxActive_b, cudaStream_t stream);

Int topk_nms(at::Tensor coords_tensor, at::Tensor heatmap_tensor, at::Tensor batch_offsets_tensor, at::Tensor semantic_labels_tensor, float R, float thres, float local_thres, int K, at::Tensor topk_idxs_tensor, at::Tensor sizes_tensor, at::Tensor k_foreground_tensor, int maxActive_f, at::Tensor k_background_tensor, int maxActive_b);

#endif //TOPK_NMS_H