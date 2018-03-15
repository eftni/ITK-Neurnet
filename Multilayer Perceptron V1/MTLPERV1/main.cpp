#include <iostream>
#include "Dataset.h"
#include "Neurnet.h"
#include "random"
#include "functional"
#include "math.h"
using namespace std;

int main()
{
    std::vector<int> layers;
    layers.push_back(16);
    layers.push_back(16);
    Neurnet net(784, layers, 10, 1, [](double x ){return 1/(1+exp(-x));}, [](double x ){return (1/(1+exp(-x))) * (1 - 1/(1+exp(-x)));});
    /*std::vector<std::random_device::result_type> seed = net.get_seed();
    std::cout << std::endl;
    for(int i = 0; i <= 7; ++i){
        std::cout << seed[i] << std::endl;
    }*/
    //net.operator<<(std::cout);
    return 0;
}
