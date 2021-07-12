//
// Created by yonatan on 12.7.2021.
//

#ifndef L2_HASH_SOFTMAX_H
#define L2_HASH_SOFTMAX_H


class Softmax:public Layer {
public:
    Softmax(int input_dim):input_dim(input_dim) {
    }
    void forward(float ** input, int input_size, float ** output) {
        for (int i = 0; i < input_size; ++i) {

            int j;
            double m, sum, constant;

            m = -INFINITY;
            for (j = 0; j < input_dim; ++j) {
                if (m < input[i][j]) {
                    m = input[i][j];
                }
            }

            sum = 0.0;
            for (j = 0; j < input_dim; ++j) {
                sum += exp(input[i][j] - m);
            }

            constant = m + log(sum);
            for (j = 0; j < input_dim; ++j) {
                output[i][j] = exp(input[i][j] - constant);
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


#endif //L2_HASH_SOFTMAX_H
