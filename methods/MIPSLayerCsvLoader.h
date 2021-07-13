//
// Created by yonatan on 12.7.2021.
//

#ifndef L2_HASH_MIPSLAYERCSVLOADER_H
#define L2_HASH_MIPSLAYERCSVLOADER_H
#include "string"
#include <iostream>
#include <fstream>
#include <sstream>
#include "helper_funcs.h"
class MIPSLayerCsvLoader {
public:
    MIPSLayerCsvLoader(const std::string& weights_path,const std::string& bias_path){
        auto weights_vec = read_csv(weights_path);
        auto bias_vec = read_csv(bias_path);
        weights = createNKMatrix(weights_vec.size(),weights_vec[0].size());
        bias = new float[bias_vec.size()];
        for (int i = 0; i < weights_vec.size(); ++i) {
            for (int j = 0; j < weights_vec[0].size(); ++j) {
                weights[i][j] = weights_vec[i][j];
            }
        }
        for (int i = 0; i < bias_vec.size(); ++i) {
            bias[i] = bias_vec[i][0];
        }

        layer = new MIPSLayer(weights_vec[0].size(),weights_vec.size(), 2, 0.99, weights, bias,10, true);
    }
    std::vector<std::vector<float >> read_csv(const std::string& path){
        std::ifstream f;
        f.open(path);
        if(!f.is_open()){
            std::cerr << "error: file open failed '" << path << "'.\n";
        }
        std::string line, val;
        std::vector<std::vector<float >> array;    /* vector of vector<int>  */

        /* string for line & value */
        while (std::getline (f, line)) {        /* read each line */
            std::vector<float> v;                 /* row vector v */
            std::stringstream s (line);         /* stringstream line */
            while (getline (s, val, ','))       /* get each value (',' delimited) */
                v.push_back(std::stof (val));  /* add to row vector */
            array.push_back(v);                /* add row vector to array */
        }
        return array;

    }
    ~MIPSLayerCsvLoader(){
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


#endif //L2_HASH_MIPSLAYERCSVLOADER_H
