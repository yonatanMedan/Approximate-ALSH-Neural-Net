#include <iostream>
#include <algorithm>
#include <cstring>
#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;
#include "Sequential.h"
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
    int    cnt       = 1;
    int topk = 80; //default number of active neurons for the first layer
    while (cnt < nargs) {
        if (strcmp(args[cnt], "-topk") == 0) {
            topk = atoi(args[++cnt]);
            if (topk < 1 || topk > 784) {
                break;
            }
        }
        cnt++;
    }
    printf("Number of active neurons for the first layer = %d\n", topk);

///////////////////////////////////////LOAD WEIGHTS///////////////////////////////////////////
    MIPSLayerCsvLoader csvLayerLoader = MIPSLayerCsvLoader("./data/FFNNMinst/layer1_weights.csv","./data/FFNNMinst/layer1_bias.csv",topk,0.99);
    MIPSLayerCsvLoader csvLayer2Loader = MIPSLayerCsvLoader("./data/FFNNMinst/layer2_weights.csv","./data/FFNNMinst/layer2_bias.csv",10);
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
    auto dataset = mnist::read_dataset<std::vector, std::vector, float, uint8_t>("./data/Mnist/test");
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
	return 0;
}