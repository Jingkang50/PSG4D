#include "otsu.h"

void otsu(at::Tensor input_tensor, at::Tensor output_tensor, int step, int inst_num, int pts_num){
    int *input = input_tensor.data<int>();
    float *output = output_tensor.data<float>();

    int hist[inst_num * step] = {0}; 
    get_hist_cuda(input, hist, step, inst_num, pts_num);
    
    for(Int idx_inst=0; idx_inst<inst_num; idx_inst++){
        float max_sigma = 0.0;
        float thre = 0.0;
        for(Int target=0; target<step; target++ ){
            int higher = 0;
            int higher_num = 0;
            int lower= 0;
            int lower_num = 0;
            for(Int sample=0; sample<step; sample++){
                if(sample < target){
                    lower += sample * hist[idx_inst*step+sample];
                    lower_num += hist[idx_inst*step+sample];
                }
                else if(sample >= target){
                    higher += sample * hist[idx_inst*step+sample];
                    higher_num += hist[idx_inst*step+sample];
                }
            } 

            if(higher_num*lower_num != 0){
                float sigma = float(higher)/higher_num-float(lower)/lower_num;
                sigma = sigma * sigma;
                sigma = higher_num*lower_num * sigma;
                if (sigma > max_sigma){
                    max_sigma = sigma;
                    output[idx_inst] = float(target) / step;
                }
            }
        }
    }
}