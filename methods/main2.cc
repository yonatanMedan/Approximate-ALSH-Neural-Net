#include <iostream>
#include <algorithm>
#include <cstring>
#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;
#include "Sequential.h"
#include "def.h"
#include "util.h"
#include "amips.h"
#include "pre_recall.h"
#include "MIPSLayerLoader.h"
#include "Relu.h"
#include "Softmax.h"
#include "mnist-master/include/mnist/mnist_reader.hpp"
#include "MIPSLayerCsvLoader.h"
using namespace mips;


// -----------------------------------------------------------------------------
int main(int nargs, char **args)
{
//	srand(6);						// srand((unsigned) time(NULL));
	int    qn        = 1;			// number of query objects
	int    d         = LAYER_DIM;			// dimensionality

	float  **query   = NULL;		// query objects
///////////////////////////////////////LOAD WEIGHTS///////////////////////////////////////////
    MIPSLayerCsvLoader csvLayerLoader = MIPSLayerCsvLoader("../data/FFNNMinst/layer1_weights.csv","../data/FFNNMinst/layer1_bias.csv");
///////////////////////////////////////LOAD DATASET//////////////////////////////////////////
    auto dataset = mnist::read_dataset<std::vector, std::vector, float, uint8_t>("../data/Mnist/test");
    std::cout << "Nbr of training images = " << dataset.training_images.size() << std::endl;
    std::cout << "Nbr of training labels = " << dataset.training_labels.size() << std::endl;
    std::cout << "Nbr of test images = " << dataset.test_images.size() << std::endl;
    std::cout << "Nbr of test labels = " << dataset.test_labels.size() << std::endl;

    query =new float *[qn];
    query[0] = new float[d];
    float ** output = new float *[qn];
    for (int i = 0; i < qn; ++i) {
        output[i] = new float[LAYER_SIZE];
    }
    for (int i = 0; i < d; ++i) {
        query[0][i] = i;
    }
    MIPSLayerLoader layerLoader = MIPSLayerLoader();
    MIPSLayer * layer = layerLoader.getLayer();
    Relu * relu_layer = new Relu(layer->getOutputDim());
    Softmax * softmax_layer = new Softmax(relu_layer->getOutputDim());
    Sequential sequentialNN = Sequential();
    sequentialNN.addLayer(layer);
    sequentialNN.addLayer(relu_layer);
//    sequentialNN.addLayer(softmax_layer);
    auto start = high_resolution_clock::now();

    sequentialNN.forward(query,qn,output);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Time taken by function: "
         << duration.count() << " microseconds" << endl;

    for (int i = 0; i < qn; ++i) {
        printf("[");

        for (int j = 0; j < LAYER_SIZE; ++j) {
            printf("%f,",output[i][j]);
        }
        printf("]\n");
    }


//    calculate_k_norm(d,query[0],norm_q[0]);
//    int TOP_K = 20;
//    MaxK_List* list = new MaxK_List(TOP_K);
//    lsh->kmip(TOP_K,(const float *)query[0],(const float *)norm_q[0],list);
//    list->output_dim();




	// -------------------------------------------------------------------------
	//  release space
//	// -------------------------------------------------------------------------
//	for (int i = 0; i < n; ++i) {
//		delete[] data[i];
//		delete[] norm_d[i];
//	}
//	delete[] data;
//	delete[] norm_d;
//
    for (int i = 0; i < qn; ++i) {
        delete[] query[i];
        delete[] output[i];
    }
    delete[] query;
    delete[] output;

	return 0;
}