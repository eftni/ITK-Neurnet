#include <iostream>
#include "Dataset.h"
#include "Neurnet.h"
#include "random"
#include "functional"
#include "math.h"
#include "chrono"
#include "KernelFunctor.h"

int main()
{
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<Layer> layers;
    layers.push_back(Layer(784, hyp_tan));
    layers.push_back(Layer(16, hyp_tan));
    layers.push_back(Layer(16, hyp_tan));
    layers.push_back(Layer(10, hyp_tan));
    KernelFunctor forprop_kernel("forprop.cl");
    KernelFunctor delta_kernel("calc_deltas.cl");
    KernelFunctor backprop_kernel("backprop.cl");
    /*while(true){
        Neurnet net(layers, 0.2);
        int epochs = 10;
        for(int i = 1; i <= epochs; ++i){
            std::cout << "Epoch: " << i << std::endl;
            Dataset training(".\\Data\\train-images.idx3-ubyte", ".\\Data\\train-labels.idx1-ubyte");
            net.train_net(training, 1000);
        }
        Dataset testing(".\\Data\\t10k-images.idx3-ubyte", ".\\Data\\t10k-labels.idx1-ubyte");
        net.test_net(testing);
    }*/
    start = std::chrono::high_resolution_clock::now();
    Neurnet net(layers, 0.2, 1000, forprop_kernel, delta_kernel, backprop_kernel);
    int epochs = 10;
    for(int i = 1; i <= epochs; ++i){
        std::cout << "Epoch: " << i << std::endl;
        Dataset training(".\\Data\\train-images.idx3-ubyte", ".\\Data\\train-labels.idx1-ubyte");
        net.train_net(training);
    }
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now()-start).count();
    std::cout << elapsed/epochs << "s" << std::endl;
    Dataset testing(".\\Data\\t10k-images.idx3-ubyte", ".\\Data\\t10k-labels.idx1-ubyte");
    net.test_net(testing);
    return 0;
}
