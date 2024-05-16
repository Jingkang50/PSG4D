#ifndef CAND_MERGING
#define CAND_MERGING
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THC.h>

#include "../datatype/datatype.h"

void cand_merging(at::Tensor score_map_tensor, at::Tensor ins_map_tensor, int ins_num, float thres);

#endif //CAND_MERGING