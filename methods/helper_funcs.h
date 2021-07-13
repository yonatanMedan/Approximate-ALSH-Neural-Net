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
};


#endif //L2_HASH_HELPER_FUNCS_H
