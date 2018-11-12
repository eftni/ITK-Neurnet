#include <iostream>
#include "include/Dataset.h"
#include "include/Neurnet.h"
#include "random"
#include "functional"
#include "math.h"
#include "chrono"
#include "include/KernelFunctor.h"

int main()
{
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<Layer> layers;
    layers.push_back(Layer(784, identity));
    layers.push_back(Layer(16, hyp_tan));
    layers.push_back(Layer(16, hyp_tan));
    layers.push_back(Layer(10, hyp_tan));
    KernelFunctor forprop_kernel("forprop.cl");
    KernelFunctor delta_kernel("calc_deltas.cl", forprop_kernel);
    KernelFunctor backprop_kernel("backprop.cl", forprop_kernel);
    Dataset training(".\\Data\\train-images.idx3-ubyte", ".\\Data\\train-labels.idx1-ubyte");
    Neurnet net(layers, 10, 1000, forprop_kernel, delta_kernel, backprop_kernel);
    net.GPUtest(training.get_im(), training.get_label());
    return 0;
}
