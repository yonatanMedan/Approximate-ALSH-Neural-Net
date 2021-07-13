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
#include "helper_funcs.h"
using namespace mips;


// -----------------------------------------------------------------------------
int main(int nargs, char **args)
{
//	srand(6);						// srand((unsigned) time(NULL));
	int    qn        = 1;			// number of query objects
	int    d         = LAYER_DIM;			// dimensionality

	float  **query   = NULL;		// query objects
///////////////////////////////////////LOAD WEIGHTS///////////////////////////////////////////
    MIPSLayerCsvLoader csvLayerLoader = MIPSLayerCsvLoader("../data/FFNNMinst/layer1_weights.csv","../data/FFNNMinst/layer1_bias.csv",80,0.99);
    MIPSLayerCsvLoader csvLayer2Loader = MIPSLayerCsvLoader("../data/FFNNMinst/layer2_weights.csv","../data/FFNNMinst/layer2_bias.csv",10);
///////////////////////////////////////Initialize NeuralNet//////////////////////////////////
    MIPSLayer * layer1 = csvLayerLoader.getLayer();
    MIPSLayer * layer2 = csvLayer2Loader.getLayer();
    Relu * relu_layer1 = new Relu(layer1->getOutputDim());
    Softmax * softmax_layer = new Softmax(layer2->getOutputDim());
    Sequential sequentialNN1 = Sequential();
    sequentialNN1.addLayer(layer1);
    sequentialNN1.addLayer(relu_layer1);
    sequentialNN1.addLayer(layer2);
    sequentialNN1.addLayer(softmax_layer);
///////////////////////////////////////LOAD DATASET//////////////////////////////////////////
    auto dataset = mnist::read_dataset<std::vector, std::vector, float, uint8_t>("../data/Mnist/test");
    std::cout << "Nbr of training images = " << dataset.training_images.size() << std::endl;
    std::cout << "Nbr of training labels = " << dataset.training_labels.size() << std::endl;
    std::cout << "Nbr of test images = " << dataset.test_images.size() << std::endl;
    std::cout << "Nbr of test labels = " << dataset.test_labels.size() << std::endl;
    float MIPS_MEAN =0.1307;
    float MIPS_STD = 0.3081;
    float ** test_data = createNKMatrix(dataset.test_images.size(),dataset.test_images[0].size());
    for (int i = 0; i < dataset.test_images.size(); ++i) {
        for (int j = 0; j < dataset.test_images[0].size(); ++j) {
            test_data[i][j] = ((dataset.test_images[i][j]/255)-MIPS_MEAN)/MIPS_STD;//devide by max pixel and nomalize
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////APROXIMATE NN INFERENCE////////////////////////////////
    int NUM_CLASSES = 10;
    float ** MINT_output = createNKMatrix(dataset.test_images.size(),NUM_CLASSES);
    auto start = high_resolution_clock::now();
    sequentialNN1.forward(test_data,dataset.test_images.size(),MINT_output);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Time taken by for all predictions: "
         << duration.count() << " microseconds" << endl;

    cout << "Average Time for one predictions: "
         << duration.count()/dataset.test_images.size() << " microseconds" << endl;
    int * predictions = new int[dataset.test_images.size()];
    getPredictions(MINT_output,dataset.test_images.size(),NUM_CLASSES,predictions);

    float num_right = 0;
    float num_examples = dataset.test_images.size();
    for (int i = 0; i < num_examples; ++i) {
        if(predictions[i]==dataset.test_labels[i]){
            num_right+=1;
        }
    }
    float accuracy = num_right/num_examples;
    cout << "Accuracy for Approximate NN model: "<<accuracy<<endl;
    delete[] predictions;
    destroyNKMatrix(MINT_output,dataset.test_images.size());

///////////////////////////////////////////////////////////////////////////////////////
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
    Sequential sequentialNN = Sequential();
    sequentialNN.addLayer(layer);
    sequentialNN.addLayer(relu_layer);
//    sequentialNN.addLayer(softmax_layer);
    start = high_resolution_clock::now();

    sequentialNN.forward(query,qn,output);
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
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
//        delete[] query[i];
        delete[] output[i];
    }
//    delete[] query;
    delete[] output;
	return 0;
}