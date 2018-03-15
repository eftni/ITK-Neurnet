#include "Neurnet.h"
#include "random"
#include "iostream"
#include "functional"

void warmup(std::function<double()>& randgen){
    for(int i = 0; i <= 100000; ++i){
        randgen();
    }
}

std::function<double()> get_randgen(std::vector<std::random_device::result_type>& seeds){
    std::random_device r;
    for(int i = 0; i <= 7; ++i){    ///RANDOM DEVICE IMPLEMENTED INCORRECTLY: USE BOOST
        seeds[i] = r();
        //std::cout << seeds[i] << std::endl;
    }
    std::seed_seq s{seeds[0],seeds[1],seeds[2],seeds[3],seeds[4],seeds[5],seeds[6],seeds[7]};
    std::function<double()> randgen1 = std::bind(std::uniform_real_distribution<double>(-1,1), std::mt19937(s));
    warmup(randgen1);
    return randgen1;
}


Neurnet::Neurnet(int input, std::vector<int> layer_count, int output, double learnrate, std::function<double(double)> activator, std::function<double(double)> derivative) :
learning_rate(learnrate),
act_func(activator),
act_func_derivative(derivative),
randgen_seeds(8, 0),
layers(layer_count.size()+2, std::vector<double>(1,0)),
weights(layer_count.size()+1, std::vector<std::vector<double>>(1, std::vector<double>(1,0)))
{
    std::function<double()> randgen = get_randgen(randgen_seeds);
    layers[0] = std::vector<double>(input,0);       ///ELIMINATE INPUT LAYER - ELIMINATE LAYERS
    for(int i = 0; i < layer_count.size(); ++i){
        layers[i+1] = std::vector<double>(layer_count[i],0);
    }
    layers[layers.size()-1] = std::vector<double>(output,0);
    for(int z = 0; z < layers.size()-1; ++z){
        weights[z] = std::vector<std::vector<double>>(layers[z+1].size(), std::vector<double>(layers[z].size(),0));
        for(int y = 0; y <weights[z].size(); ++y){
            for(int x = 0; x < weights[z][y].size(); ++x){
                weights[z][y][x] = randgen();
                //std::cout << randgen() << std::endl;
            }
        }
    }
}

Neurnet::~Neurnet()
{
    //dtor
}


int Neurnet::forprop(std::vector<std::vector<uint8_t>> image){

}

/*std::ostream& Neurnet::operator<<(std::ostream& out){
    for(int y = 0; y <weights[1].size(); ++y){
        for(int x = 0; x < weights[1][y].size(); ++x){
            out << weights[1][y][x] << '|';
        }
        out << std::endl;
    }
    return out;
}*/
