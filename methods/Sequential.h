//
// Created by yonatan on 12.7.2021.
//

#ifndef L2_HASH_SEQUENTIAL_H
#define L2_HASH_SEQUENTIAL_H
#include "vector"
#include "Layer.h"
class Sequential:public Layer {
public:
    Sequential(){

    }
    void forward(float ** input, int input_size,
                 float ** output){
        float ** current_input = input;
        float ** current_output = NULL;
        for (int i = 0; i < layers.size(); ++i) {
            auto layerPtr = layers[i];

            if(i<layers.size()-1){
                current_output = createNKMatrix(input_size,layerPtr->getOutputDim());
            } else{
                current_output = output;

            }
            layerPtr->forward(current_input,input_size,current_output);
            destroyNKMatrix(current_input,input_size);
            current_input = current_output;
        }
    }
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
    void addLayer(Layer *layer){
        layers.push_back(layer);
    }

    int getInputDim(){
        if(layers.size()>0){
            return layers[0]->getInputDim();
        } else {
            return -1;
        }
    }
    int getOutputDim() {
        if(layers.size()>0){
            return layers[layers.size()-1]->getOutputDim();
        } else {
            return -1;
        }
    }
    ~Sequential(){

    }

protected:
    vector<Layer*> layers;
};


#endif //L2_HASH_SEQUENTIAL_H
