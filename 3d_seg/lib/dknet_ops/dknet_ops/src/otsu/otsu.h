#ifndef OTSU
#define OSTU
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>

#include "../datatype/datatype.h"

void otsu(at::Tensor input, at::Tensor output, int step, int inst_num, int pts_num);
void get_hist_cuda(int* input, int* hist, int step, int inst_num, int pts_num);

#endif //OTSU