#include "Neurnet.h"
#include "random"
#include "iostream"
#include "functional"
//#include "..\\matrixmath.h"
#include "time.h"
#include "stdio.h"
#include "math.h"
#include "fstream"
#include "sstream"


Neurnet::Neurnet(){

}

/**
* Warms up the Mersenne Twister by producing 100000 random numbers
* @param randgen A callable random generator
*/
void warmup(std::function<float()>& randgen){
    for(int i = 0; i <= 100000; ++i){
        randgen();
    }
}

/**
* Initializes the random generator used for assigning random weights and biases at the start of training.
* (The random generator is a Mersenne Twister using a seed sequence of 8 seeds and bound to a
* a uniform distrbution functor.)
* @param seeds A set of seeds to be used by the generator
* @return A callable object which generates random numbers of a uniform distribution.
*/
std::function<float()> get_randgen(std::vector<std::random_device::result_type>& seeds){
    std::cout << "Creating random generator" << std::endl;
    srand(time(nullptr));
    //std::random_device r;
    if(seeds == std::vector<std::random_device::result_type>(8,0)){
        for(int i = 0; i <= 7; ++i){    //RANDOM DEVICE IMPLEMENTED INCORRECTLY: USE BOOST
            //seeds[i] = r();
            //seeds[i] = rand();
            seeds[i] = i; //DEBUG - REMOVE AFTER TESTING
            //std::cout << seeds[i] << std::endl;
        }
    }
    std::seed_seq s{seeds[0],seeds[1],seeds[2],seeds[3],seeds[4],seeds[5],seeds[6],seeds[7]};
    std::function<float()> randgen1 = std::bind(std::uniform_real_distribution<float>(-1,1), std::mt19937(s));
    warmup(randgen1);
    return randgen1;
}

void Neurnet::create_buffers(cl::Context c){
    input_buffers.emplace_back(c, CL_MEM_READ_WRITE, 0);
    output_buffers.emplace_back(c, CL_MEM_READ_WRITE, sizeof(float)*n_layers[0].n_number);
    for(size_t i = 1; i < n_layers.size(); ++i){
        input_buffers.emplace_back(c, CL_MEM_READ_WRITE, sizeof(float)*n_layers[i].n_number);
        delta_buffers.emplace_back(c, CL_MEM_READ_WRITE, sizeof(float)*n_layers[i].n_number);
        output_buffers.emplace_back(c, CL_MEM_READ_WRITE, sizeof(float)*n_layers[i].n_number);
        w_buffers.emplace_back(c, CL_MEM_READ_WRITE, sizeof(float)*(n_layers[i].n_number)*n_layers[i-1].n_number);
        w_update_buffers.emplace_back(c, CL_MEM_READ_WRITE, sizeof(float)*(n_layers[i].n_number)*n_layers[i-1].n_number);
    }
}


Neurnet::Neurnet(std::vector<Layer> layers, float learnrate, size_t batch, KernelFunctor fp_ker, KernelFunctor delta_ker, KernelFunctor bp_ker, std::vector<std::random_device::result_type> rs) :
learning_rate(learnrate),
batch_size(batch),
randgen_seeds(rs),
weights(layers.size()-1, std::vector<float>(1, 0)),
biases(layers.size(), std::vector<float>(1, 0)),
n_layers(layers),
hit(0),
miss(0),
logfile("Log001.txt")
{
    std::function<float()> randgen = get_randgen(randgen_seeds);
    std::cout << "Generating weights" << std::endl;
    for(size_t y = 0; y < layers.size()-1; ++y){
        weights[y] = std::vector<float>(layers[y].n_number*layers[y+1].n_number, 0);
        for(size_t x = 0; x < weights[y].size(); ++x){
            weights[y][x] = randgen();
        }
    }
    forprop_kernel = fp_ker;
    delta_kernel = delta_ker;
    backprop_kernel = bp_ker;
    create_buffers(forprop_kernel.get_context());
    float zero = 0;
    for(size_t i = 0; i < weights.size(); ++i){
        forprop_kernel.c_queue.enqueueWriteBuffer(w_buffers[i], CL_FALSE, 0, sizeof(float)*weights[i].size(), &weights[i][0]);
        forprop_kernel.c_queue.enqueueFillBuffer(w_update_buffers[i], &zero, 0, sizeof(float));
    }
    /*for(size_t i = 0; i < layers.size(); ++i){
        biases[i] = std::vector<float>(layers[i].n_number, 0);
        for(size_t j = 0; j < layers[i].n_number; ++j){
            biases[i][j] = randgen();
        }
    }*/
    std::cout << "Weights generated: Starting training" << std::endl;
}

Neurnet::~Neurnet()
{
    //dtor
}

template<typename T>
void operator-=(std::vector<T>& w, const std::vector<T>& update){
    for(size_t i = 0; i < w.size(); ++i){
        w[i] -= update[i];
    }
}

template<typename T>
std::vector<T> operator*(const std::vector<T>& w, const float& f){
    std::vector<T> temp(w);
    for(size_t i = 0; i < w.size(); ++i){
        temp[i] = w[i]*f;
    }
    return temp;
}

template<typename T>
std::vector<T> operator/(const std::vector<T>& w, const float& f){
    std::vector<T> temp(w);
    for(size_t i = 0; i < w.size(); ++i){
        temp[i] = w[i]/f;
    }
    return temp;
}

void Neurnet::GPUtest(std::vector<uint8_t> im, uint8_t lab){
    std::vector<float> temp(im.begin(), im.end());
    temp = temp/255;
    forprop_kernel.c_queue.enqueueWriteBuffer(output_buffers[0], CL_TRUE, 0, sizeof(float)*temp.size(), &temp[0]);
    for(float f : temp){
        std::cout << f << ' ';
    }
    std::cout << std::endl;
    std::vector<float> temp2(temp.size(), 0);
    forprop_kernel.c_queue.enqueueReadBuffer(output_buffers[0], CL_TRUE, 0, temp.size()*sizeof(float), &temp2[0]);
    for(float f : temp2){
        std::cout << f << ' ';
    }
    std::cout << std::endl;
    int i = 0;
    forprop_kernel(cl::NullRange, cl::NDRange(n_layers[i+1].n_number, batch_size), cl::NullRange, output_buffers[i], n_layers[i].n_number, w_buffers[i], n_layers[i].n_number, input_buffers[i+1], n_layers[i+1].n_number, n_layers[i+1].activator, output_buffers[i+1]);
    std::vector<float> forprop_result(16,0);
    forprop_kernel.c_queue.enqueueReadBuffer(input_buffers[i+1], CL_TRUE, 0, forprop_result.size()*sizeof(float), &forprop_result[0]);
    for(float f : forprop_result){
        std::cout << f << ' ';
    }
    std::cout << std::endl;
    forprop_kernel.c_queue.enqueueReadBuffer(output_buffers[i+1], CL_TRUE, 0, forprop_result.size()*sizeof(float), &forprop_result[0]);
    for(float f : forprop_result){
        std::cout << f << ' ';
    }
    std::cout << std::endl;
}
