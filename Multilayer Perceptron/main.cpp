#include <iostream>
#include "Dataset.h"
#include "Neurnet.h"
#include "random"
#include "functional"
#include "math.h"

int main()
{
    std::vector<Layer> layers;
    layers.push_back(Layer(784, hyp_tan));
    layers.push_back(Layer(16, hyp_tan));
    layers.push_back(Layer(16, hyp_tan));
    layers.push_back(Layer(10, hyp_tan));
    while(true){
        Neurnet net(layers, 0.2);
        int epochs = 10;
        for(int i = 1; i <= epochs; ++i){
            std::cout << "Epoch: " << i << std::endl;
            Dataset training(".\\Data\\train-images.idx3-ubyte", ".\\Data\\train-labels.idx1-ubyte");
            net.train_net(training, 1000);
        }
        Dataset testing(".\\Data\\t10k-images.idx3-ubyte", ".\\Data\\t10k-labels.idx1-ubyte");
        net.test_net(testing);
    }
    return 0;
}
