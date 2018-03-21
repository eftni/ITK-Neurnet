#include <iostream>
#include "Dataset.h"
#include "Neurnet.h"
#include "random"
#include "functional"
#include "math.h"

using namespace std;

int main()
{
    Dataset training(".\\Data\\train-images.idx3-ubyte", ".\\Data\\train-labels.idx1-ubyte");
    Dataset testing(".\\Data\\t10k-images.idx3-ubyte", ".\\Data\\t10k-labels.idx1-ubyte");
    std::vector<int> layers;
    layers.push_back(784);
    layers.push_back(16);
    layers.push_back(16);
    layers.push_back(10);
    Neurnet net(layers, 1, [](double x ){return 1/(1+exp(-x));}, [](double x ){return x*(1-x);}); ///Derivative with respect to output
    net.train_net(training);
    net.test_net(testing);


    //Neurnet net(layers, 1, [](double x ){return 1/(1+exp(-x));}, [](double x ){return (1/(1+exp(-x))) * (1 - 1/(1+exp(-x)));});
    /*std::vector<std::random_device::result_type> seed = net.get_seed();
    std::cout << std::endl;
    for(int i = 0; i <= 7; ++i){
        std::cout << seed[i] << std::endl;
    }*/
    //net.operator<<(std::cout);
    return 0;
}
