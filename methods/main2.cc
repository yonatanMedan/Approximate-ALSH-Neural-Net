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
using namespace mips;


// -----------------------------------------------------------------------------
int main(int nargs, char **args)
{
//	srand(6);						// srand((unsigned) time(NULL));
//	//usage();
//
//	char   data_set[200] = "../data/Mnist/Mnist.ds";			// address of data set
////	char   query_set[200];			// address of query set
////	char   truth_set[200];			// address of ground truth file
////	char   out_path[200];			// output path
//
//	int    alg       = -1;			// which algorithm?
//	int    n         = 60000;			// number of data objects
	int    qn        = 1;			// number of query objects
	int    d         = LAYER_DIM;			// dimensionality
//	int    K         = -1;			// #tables for sign-alsh and simple-lsh
//	int    m         = -1;			// param for l2-alsh, l2-alsh2, sign-alsh
//	float  U         = -1.0f;		// param for l2-alsh, l2-alsh2, sign-alsh
//	float  nn_ratio  = 2;		// approximation ratio of ANN search
//	float  mip_ratio = 0.5;		// approximation ratio of AMIP search
//
//	float  **data    = NULL;		// data objects
	float  **query   = NULL;		// query objects
//	float  **norm_d  = NULL;		// l2-norm of data  objects
//	float  **norm_q  = NULL;		// l2-norm of query objects
//	Result **R       = NULL;		// truth set
//	float  **pre     = NULL;		// precision array
//	float  **recall  = NULL;		// recall array
//	bool   failed    = false;
//	int    cnt       = 1;
//
//	// -------------------------------------------------------------------------
//	//  read data set, query set, and ground truth file
//	// -------------------------------------------------------------------------
//	data   = new float*[n];
//	norm_d = new float*[n];
//	for (int i = 0; i < n; ++i) {
//		data[i]   = new float[d];
//		norm_d[i] = new float[NORM_K];
//	}
//	if (read_bin_data(n, d, true, data_set, data, norm_d)) exit(1);
//	// -------------------------------------------------------------------------
//	//  methods
//	// -------------------------------------------------------------------------
//    H2_ALSH *lsh = new H2_ALSH(n, d, nn_ratio, mip_ratio, (const float **) data, (const float **) norm_d);
//    lsh->display();
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