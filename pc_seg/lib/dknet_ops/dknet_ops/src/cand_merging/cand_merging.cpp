#include "cand_merging.h"
#include <algorithm>

void cand_merging(at::Tensor score_map_tensor, at::Tensor ins_map_tensor, int ins_num, float thres){
    float *score_map = score_map_tensor.data<float>();
    int* ins_map = ins_map_tensor.data<int>();
    Int sorted_idxs[ins_num * ins_num] = {0}; 

    std::sort(sorted_idxs, sorted_idxs+ins_num*ins_num,
        [&score_map](Int a,Int b){ return score_map[a]>score_map[b];} );
    
    Int p_idx = 0;
    Int idx = sorted_idxs[p_idx];
    while((score_map[idx] < 0) || (score_map[idx] > thres)){
        Int col = idx / ins_num;
        Int row = idx % ins_num;

        if((col < row) && (score_map[idx] > 0)){
            Int col_index = ins_map[col];
            Int row_index = ins_map[row];
            Int target_index = std::min(col_index, row_index);
            
            for(Int j=0; j<ins_num; j++){
                if(ins_map[j] == col_index){
                    for(Int k=0; k<ins_num; k++){
                        if(ins_map[k] == row_index){
                            score_map[j*ins_num + k] = -1.0;
                            score_map[k*ins_num + j] = -1.0;
                        }
                    }
                }
            }
            
            for(Int m=0; m<ins_num; m++){ 
                if((ins_map[m] == col_index) || (ins_map[m] == row_index)){
                    ins_map[m] = target_index;
                }
            }
        }

        p_idx ++;
        idx = sorted_idxs[p_idx];
    }
}