//
// Created by yonatan on 13.7.2021.
//

#ifndef L2_HASH_HELPER_FUNCS_H
#define L2_HASH_HELPER_FUNCS_H


namespace mips {
    float ** createNKMatrix(int N_rows,int K_cols){
        float ** output = new float *[N_rows];
        for (int i = 0; i < N_rows; ++i) {
            output[i] = new float[K_cols];
        }
        return output;
    }
    void destroyNKMatrix(float **matrix, int N_rows){
        for (int i = 0; i < N_rows; ++i) {
            delete[] matrix[i];
        }
        delete[] matrix;
    }
    void getPredictions(float ** nn_output,int num_examples,int num_classes ,int * predictions){
        for (int i = 0; i < num_examples; ++i) {
            int class_max_idx = -1;
            float class_max_prediction = -INFINITY;
            for (int j = 0; j < num_classes; ++j) {
                if(nn_output[i][j]>class_max_prediction){
                    class_max_prediction = nn_output[i][j];
                    class_max_idx = j;
                }
                predictions[i] = class_max_idx;
            }
        }
    }
};


#endif //L2_HASH_HELPER_FUNCS_H
