#include <iostream>
#include "Dataset.h"
#include "Neurnet.h"
#include "random"
#include "functional"
#include "math.h"
#include "Layer.h"

using namespace std;

int main()
{
    //Dataset training(".\\Data\\train-images.idx3-ubyte", ".\\Data\\train-labels.idx1-ubyte");
    //Dataset testing(".\\Data\\t10k-images.idx3-ubyte", ".\\Data\\t10k-labels.idx1-ubyte");
    std::vector<Layer> layers;
    //layers.push_back(Layer(784, [](double x){return x;}, [](double x){return 1;}, 1));
    //layers.push_back(Layer(784, [](double x){return 1/(1+exp(-x));}, [](double x){return x*(1-x);}, 0));
    layers.push_back(Layer(784, [](double x){return tanh(x);}, [](double x){return 1-pow(x,2);}, 0));
    //layers.push_back(Layer(16, [](double x){return 1/(1+exp(-x));}, [](double x){return x*(1-x);}, 0));
    //layers.push_back(Layer(16, [](double x){return 1/(1+exp(-x));}, [](double x){return x*(1-x);}, 0));
    layers.push_back(Layer(16, [](double x){return tanh(x);}, [](double x){return 1-pow(x,2);}, 0));
    layers.push_back(Layer(16, [](double x){return tanh(x);}, [](double x){return 1-pow(x,2);}, 0));
    layers.push_back(Layer(10, [](double x){return tanh(x);}, [](double x){return 1-pow(x,2);}, 0));
    Neurnet net(layers, 1);
    int epochs = 1;
    for(int i = 0; i < epochs; ++i){
        std::cout << "Epoch: " << i << std::endl;
        Dataset training(".\\Data\\train-images.idx3-ubyte", ".\\Data\\train-labels.idx1-ubyte");
        net.train_net(training, 1000);
    }
    Dataset testing(".\\Data\\t10k-images.idx3-ubyte", ".\\Data\\t10k-labels.idx1-ubyte");
    net.test_net(testing);
    return 0;
}
