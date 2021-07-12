//
// Created by yonatan on 11.7.2021.
//

#ifndef L2_HASH_MIPSLAYERLOADER_H
#define L2_HASH_MIPSLAYERLOADER_H
#include "MIPSLayer.h"


using namespace mips;
namespace mips{
    int LAYER_SIZE = 512;
    int LAYER_DIM = 100;
}

class MIPSLayerLoader {
public:
    MIPSLayerLoader(){

        this->weights = new float *[LAYER_SIZE];
        this->bias = new float[LAYER_SIZE];

        for (int i = 0; i < LAYER_SIZE; ++i) {
            weights[i] = new float [LAYER_DIM];
            for (int j = 0; j < LAYER_DIM; ++j) {
                if(i>12 && i<23){
                    weights[i][j] = 0.5;
                } else if(i>25 && i<36){
                    weights[i][j] = -0.5;
                }else{
                    weights[i][j] = 0;
                }
            }
        }
        for (int i = 0; i < LAYER_SIZE; ++i) {
            this->bias[i] = 1;
        }
        layer = new MIPSLayer(LAYER_SIZE,LAYER_DIM, 2, 0.99, weights, bias,10, true);

    };
    ~MIPSLayerLoader(){
        for (int i = 0; i < LAYER_SIZE; ++i) {
            delete[] weights[i];
        }
        delete[] weights;
        delete layer;
    };
    MIPSLayer * getLayer(){
        return layer;
    }
protected:
    float ** weights;
    float * bias;
    MIPSLayer * layer;


};


#endif //L2_HASH_MIPSLAYERLOADER_H
