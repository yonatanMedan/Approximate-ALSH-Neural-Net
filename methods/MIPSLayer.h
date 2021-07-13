//
// Created by yonatan on 11.7.2021.
//

#ifndef H2_HASH_MIPSLAYER_H
#define H2_HASH_MIPSLAYER_H
#include "h2_alsh.h"
#include "util.h"
#include "array"
#include "Layer.h"
using namespace mips;
class MIPSLayer :public Layer {
public:
    MIPSLayer(int output_dim, int input_dim, float  nn_ratio, float  mip_ratio, float ** weights,float * bias, int top_k , bool find_neg=false):
    output_dim(output_dim), dim(input_dim), nn_ratio(nn_ratio), mip_ratio(mip_ratio), weights(weights),bias(bias), top_k(top_k), find_neg(find_neg){
        norm_w = new float*[output_dim];
        for (int i = 0; i < output_dim; ++i) {
            norm_w[i] = new float[NORM_K];
            calculate_k_norm(input_dim, weights[i], norm_w[i]);

        }
        this->lsh = new H2_ALSH(output_dim, input_dim, nn_ratio, mip_ratio, (const float **) weights, (const float **) norm_w);

    }
    int getInputDim(){
        return dim;
    }
    int getOutputDim(){
        return output_dim;
    }
    virtual ~MIPSLayer(){
        for (int i = 0; i < output_dim; ++i) {
            delete[] norm_w[i];
            delete[] weights[i];
        }
        delete[] bias;
        delete[] norm_w;
        delete[] weights;
        delete lsh;

    }
    void make_negative(float * input,int input_dim){
        for (int i = 0; i < input_dim; ++i) {
            input[i] = -input[i];
        }
    }
    void forward( float ** input,int input_size,
                  float ** output){
        //delete me
        float ** norm_q = new float *[input_size];

        for (int i = 0; i < input_size; ++i) {
            auto start = high_resolution_clock::now();

            norm_q[i] = new float[NORM_K];
            for (int j = 0; j < getOutputDim(); ++j) {
                output[i][j] = bias[j];
            }
//            memset(output[i],0.0f,getOutputDim()*sizeof(float));
            calculate_k_norm(getInputDim(),input[i],norm_q[i]);
            MaxK_List* list = new MaxK_List(top_k);
            lsh->kmip(top_k,(const float *)input[i],(const float *)norm_q[i],list);
            for (int j = 0; j < top_k; ++j) {
                output[i][list->ith_id(j)-1] += list->ith_key(j);
//                printf("neuron: %d activated with product of: %f + bias of %f\n",list->ith_id(j)-1, list->ith_key(j), bias[list->ith_id(j)-1]);

            }
            delete list;
            if(find_neg){
                make_negative(input[i],getInputDim());
                MaxK_List* list_neg = new MaxK_List(top_k);
                lsh->kmip(top_k,(const float *)input[i],(const float *)norm_q[i],list_neg);
                for (int j = 0; j < top_k; ++j) {
                    output[i][list_neg->ith_id(j)-1] += -list_neg->ith_key(j);
                }
                delete list_neg;
            }
            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<microseconds>(stop - start);
//            printf("duration: %d",duration.count());


        }

        //clean up
        for (int i = 0; i < input_size; ++i) {
            delete[] norm_q[i];
        }
        delete[] norm_q;
    }

protected:
    int output_dim;
    int dim;
    float  nn_ratio;		// approximation ratio of ANN search arround 2
    float  mip_ratio;       // 0.5-0.99
    float ** weights;
    float * bias;
    float ** norm_w;
    int top_k;
    H2_ALSH * lsh;
    bool find_neg;


};


#endif //H2_HASH_MIPSLAYER_H
