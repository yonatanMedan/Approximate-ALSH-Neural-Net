//
// Created by yonatan on 12.7.2021.
//

#ifndef L2_HASH_RELU_H
#define L2_HASH_RELU_H
#include "Layer.h"

class Relu:public Layer{
public:
    Relu(int input_dim):input_dim(input_dim) {
    }
    void forward (float ** input, int input_size,
                  float ** output){
        int input_dim = getInputDim();
        for (int i = 0; i < input_size; ++i) {
            for (int j = 0; j < input_dim; ++j) {
                if(input[i][j]<0){
                    output[i][j] = 0;
                } else{
                    output[i][j] = input[i][j];
                }
            }
        }
    }
    int getInputDim(){
        return input_dim;
    }
    int getOutputDim(){
        return input_dim;
    }

protected:
    int input_dim;
};


#endif //L2_HASH_RELU_H
