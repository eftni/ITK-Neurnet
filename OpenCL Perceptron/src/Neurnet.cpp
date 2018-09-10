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
            seeds[i] = rand();
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
    output_buffers.emplace_back(c, CL_MEM_READ_WRITE, sizeof(float)*n_layers[0].n_number*batch_size);
    for(size_t i = 1; i < n_layers.size(); ++i){
        input_buffers.emplace_back(c, CL_MEM_READ_WRITE, sizeof(float)*n_layers[i].n_number*batch_size);
        delta_buffers.emplace_back(c, CL_MEM_READ_WRITE, sizeof(float)*n_layers[i].n_number*batch_size);
        output_buffers.emplace_back(c, CL_MEM_READ_WRITE, sizeof(float)*n_layers[i].n_number*batch_size);
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
std::vector<T> operator/(std::vector<T>& w, float d){
    for(size_t i = 0; i < w.size(); ++i){
        w[i] = w[i]/d;
    }
    return w;
}

void Neurnet::forprop(std::vector<uint8_t> images){
    std::vector<std::vector<float>> inputs;
    std::vector<std::vector<float>> outputs;
    std::vector<float> temp(images.begin(), images.end());
    temp = temp/255; //Input normalization
    //temp += biases[0];
    forprop_kernel.c_queue.enqueueWriteBuffer(output_buffers[0], CL_TRUE, 0, sizeof(float)*temp.size(), &temp[0]);
    for(size_t i = 0; i < weights.size(); ++i){
        forprop_kernel(cl::NullRange, cl::NDRange(n_layers[i+1].n_number, batch_size), cl::NullRange, output_buffers[i], n_layers[i].n_number, w_buffers[i], n_layers[i].n_number, input_buffers[i+1], n_layers[i+1].n_number, n_layers[i+1].activator, output_buffers[i+1]);
    }
    /*std::vector<float> test(n_layers[1].n_number, 0);
    forprop_kernel.c_queue.enqueueReadBuffer(output_buffers[1], CL_TRUE, 0, sizeof(float)*test.size(), &test[0]);
    for(float f : test){
        std::cout << f << ' ';
    }*/
}

void Neurnet::calc_deltas(std::vector<uint8_t> targets){
    cl::Buffer dummy(delta_kernel.get_context(), CL_MEM_READ_WRITE, sizeof(float));
    for(size_t i = 0; i < n_layers.size()-1; ++i){
        if(i == 0){
            cl::Buffer target_buffer(delta_kernel.get_context(), CL_MEM_READ_WRITE, sizeof(float));
            delta_kernel.c_queue.enqueueWriteBuffer(target_buffer, CL_TRUE, 0, sizeof(char)*targets.size(), &targets[0]);
            delta_kernel(cl::NullRange, cl::NDRange(n_layers.back().n_number, batch_size), cl::NullRange, 0, input_buffers.back(), output_buffers.back(), w_buffers.back(), 0, target_buffer,
                         n_layers.back().activator, dummy, 0, delta_buffers.back());
        }else{
            size_t index = n_layers.size()-(1+i);
            delta_kernel(cl::NullRange, cl::NDRange(n_layers[index].n_number, batch_size), cl::NullRange, 1, input_buffers[index], output_buffers[index], w_buffers[index], n_layers[index].n_number, dummy, //Check wsize
                         n_layers[index].activator, delta_buffers[index], n_layers[index+1].n_number, delta_buffers[index-1]);  //Check w_buffers indexing
        }
    }
}

void Neurnet::backprop(std::vector<uint8_t> targets){
    calc_deltas(targets);
    for(size_t i = 0; i < weights.size(); ++i){
        backprop_kernel(cl::NullRange, cl::NDRange(n_layers[i].n_number, n_layers[i+1].n_number, batch_size), cl::NullRange, delta_buffers[i], output_buffers[i], n_layers[i].n_number, w_update_buffers[i], n_layers[i+1].n_number, learning_rate);
    }
}

/**
* Checks if the output of a forward pass corresponds to the expected output.
* @param target The expected output
* @param actual The output of a forward pass
* @return True if the output matches the target, false otherwise
*/
size_t out_check(std::vector<uint8_t> targets, std::vector<float> actual, size_t batch_size){
    size_t hit = 0;
    for(size_t b = 0; b < batch_size; ++b){
        size_t top = 0;
        float maxi = 0;
        for(size_t i = 0; i < actual.size(); ++i){
            if(actual[b + i] > maxi){
                maxi = actual[i];
                top = i;
            }
        }
        if(targets[b] == top){
            ++hit;
        }
    }
    return hit;
}

/**
* Calculates the squared total error of a forward pass.
* (The error is halved to simplify derivation)
* @param target The expected output
* @param actual The output of a forward pass
* @return The total error.
*/
float calc_total_error(std::vector<uint8_t> targets, std::vector<float> actual, size_t batchsize){;
    float err_tot = 0;
    for(size_t b = 0; b < batchsize; ++b){
        for(size_t i = 0; i < actual.size(); ++i){
            if(targets[b] == i){
                err_tot += (pow(1-actual[i],2))/2;
            }else{
                err_tot += (pow(0-actual[i],2))/2;
            }
        }
    }
    return err_tot;
}

void Neurnet::single_pass(std::pair<std::vector<uint8_t>, std::vector<uint8_t>> im_lab){
    forprop(im_lab.first);
    std::vector<float> actual(n_layers.back().n_number*batch_size, 0);
    forprop_kernel.c_queue.enqueueReadBuffer(output_buffers.back(), CL_TRUE, 0, sizeof(float)*n_layers.back().n_number*batch_size, &actual[0]);
    int correct = out_check(im_lab.second, actual, batch_size);
    hit += correct;
    miss += batch_size-correct;
}

float Neurnet::train_pass(std::pair<std::vector<uint8_t>, std::vector<uint8_t>> im_lab){
    forprop(im_lab.first);
    std::vector<float> actual(n_layers.back().n_number, 0);
    backprop(im_lab.second);
    /*for(size_t i = 0; i < ins_outs.second.back().size(); ++i){
        logfile << ins_outs.second.back()[i] << ' ' << target[i] << ' ' << (pow(target[i]-ins_outs.second.back()[i],2))/2 << std::endl;
    }
    logfile << "--------" << std::endl;*/
    forprop_kernel.c_queue.enqueueReadBuffer(output_buffers.back(), CL_TRUE, 0, sizeof(float)*actual.size(), &actual[0]);
    return calc_total_error(im_lab.second, actual, batch_size);
}

void Neurnet::train_net(Dataset& training){
    logfile << "------------Training error values------------" << std::endl;
    //int index = 0;
    while(training.check_over()){
        float err_tot_sum = train_pass(training.load_batch(batch_size));
        float zero = 0;
        for(size_t i = 0; i < w_update_buffers.size(); ++i){
            std::vector<float> w_update(weights[i].size(), 0);
            forprop_kernel.c_queue.enqueueReadBuffer(w_update_buffers[i], CL_TRUE, 0, sizeof(float)*w_update.size(), &w_update[0]);
            weights[i] -= w_update/batch_size;
            forprop_kernel.c_queue.enqueueFillBuffer(w_update_buffers[i], &zero, 0, sizeof(float));
            forprop_kernel.c_queue.enqueueWriteBuffer(w_buffers[i], CL_FALSE, 0, sizeof(float)*weights[i].size(), &weights[i][0]);
        }
        std::cout << training.get_index()-1 << '/' << 60000 << std::endl;
        std::cout << "Total error:" << err_tot_sum/batch_size << std::endl;
        err_tot_sum = 0;
    }
}

std::ostream& operator<<(std::ostream& out, std::vector<float> v){
    for(float f : v){
        out << f << ' ';
    }
    out << ';' << std::endl;
    return out;
}

void Neurnet::write_to_master(){
    std::ofstream master("master.txt");
    master << "-------Current best network:-------" << std::endl;
    master << "Percentage: " << ((float)hit/10000)*100 << std::endl;
    master << "Hit: " << hit << " Miss: " << miss << std::endl;
    master << "-------Network settings:-------" << std::endl;
    master << "Seeds:" << std::endl;
    for(std::random_device::result_type s : randgen_seeds){
        master << s << std::endl;
    }
    master << "Layers:" << std::endl;
    master << n_layers.size() << std::endl;
    for(Layer l : n_layers){
        master << l.n_number << '\t'; //Cannot handle act functions yet
    }
    master << learning_rate << std::endl;
    master << n_layers.size() << std::endl;
    for(size_t i = 0; i < weights.size(); ++i){
        master << weights[i];
    }
    master.close();
}

/**
* Reads how well the current best network performed
*/
float read_master_best(){
    std::ifstream fin("master.txt");
    std::string s;
    std::getline(fin, s);
    std::getline(fin, s, ' ');
    std::getline(fin, s);
    return atof(s.c_str());
}

void Neurnet::test_net(Dataset& testing){
    logfile << "------------Testing hit rate------------" << std::endl;
    while(testing.check_over()){
        single_pass(testing.load_batch(batch_size));
        if(testing.get_index()%500 == 0){
            std::cout << testing.get_index() << '/' << 10000 << std::endl;
        }
        testing.load_one();
    }
    std::cout << "hit: " << hit << " miss: " << miss << std::endl;
    std::cout << "ratio: " << ((float)hit/10000)*100 << '%' << std::endl;
    float best = read_master_best();
    if(((float)hit/10000)*100 > best){
        write_to_master();
    }
    logfile << "hit: " << hit << " miss: " << miss << std::endl;
    logfile << "ratio: " << ((float)hit/10000)*100 << '%' << std::endl;
    logfile.close();
}

