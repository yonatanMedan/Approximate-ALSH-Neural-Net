//
// Created by yonatan on 12.7.2021.
//

#ifndef L2_HASH_LAYER_H
#define L2_HASH_LAYER_H


class Layer {
public:
    virtual void forward(float ** input, int input_size,
            float ** output)=0;
    virtual int getInputDim()=0;
    virtual int getOutputDim()=0;
    virtual ~Layer(){};
};


#endif //L2_HASH_LAYER_H
