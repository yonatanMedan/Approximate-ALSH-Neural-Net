//
// Created by yonatan on 11.7.2021.
//

#ifndef H2_HASH_MIPSLAYER_H
#define H2_HASH_MIPSLAYER_H
#include "h2_alsh.h"
#include "util.h"
#include "array"

using namespace mips;
class MIPSLayer {
public:
    MIPSLayer(int size, int dim,float  nn_ratio,float  mip_ratio,  float ** weights, int top_k ):size(size),dim(dim),nn_ratio(nn_ratio),mip_ratio(mip_ratio),weights(weights),top_k(top_k){
        norm_w = new float*[size];
        for (int i = 0; i < size; ++i) {
            norm_w[i] = new float[NORM_K];
            calculate_k_norm(dim,weights[i],norm_w[i]);

        }
        this->lsh = new H2_ALSH(size,dim,nn_ratio,mip_ratio,(const float **) weights,(const float **) norm_w);

    }
    ~MIPSLayer(){
        for (int i = 0; i < size; ++i) {
            delete[] norm_w[i];
        }
        delete[] norm_w;
    }
    void make_negative(float * input,int input_dim){
        for (int i = 0; i < input_dim; ++i) {
            input[i] = -input[i];
        }
    }
    void Multiply( float ** input,int input_size,int input_dim,
                  float ** output,int output_dim, bool find_neg=false){
        //delete me
        float ** norm_q = new float *[input_size];

        for (int i = 0; i < input_size; ++i) {
            norm_q[i] = new float[NORM_K];
            memset(output[i],0.0f,output_dim*sizeof(float));
            calculate_k_norm(input_dim,input[i],norm_q[i]);
            MaxK_List* list = new MaxK_List(top_k);
            lsh->kmip(top_k,(const float *)input[i],(const float *)norm_q[i],list);
            for (int j = 0; j < top_k; ++j) {
                output[i][list->ith_id(j)] = list->ith_key(j);
            }
            if(find_neg){
                make_negative(input[i],input_dim);
                MaxK_List* list_neg = new MaxK_List(top_k);
                lsh->kmip(top_k,(const float *)input[i],(const float *)norm_q[i],list_neg);
                for (int j = 0; j < top_k; ++j) {
                    output[i][list_neg->ith_id(j)] = -list_neg->ith_key(j);
                }
            }

        }

        //clean up
        for (int i = 0; i < input_size; ++i) {
            delete[] norm_q[i];
        }
        delete[] norm_q;
    }

protected:
    int size;
    int dim;
    float  nn_ratio;		// approximation ratio of ANN search arround 2
    float  mip_ratio;       // 0.5-0.99
    float ** weights;
    float ** norm_w;
    int top_k;
    H2_ALSH * lsh;


};


#endif //H2_HASH_MIPSLAYER_H
