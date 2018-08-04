#include "Neurnet.h"
#include "random"
#include "iostream"
#include "functional"
#include "..\\matrixmath.h"
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

size_t sum_weight_elements(const std::vector<std::vector<std::vector<float>>>& weights){
    size_t acc = 0;
    for(std::vector<std::vector<float>> vv : weights){
        acc += vv.size()*vv[0].size();
    }
    return acc;
}

void Neurnet::create_buffers(cl::Context c){
    weight_buffer = cl::Buffer(c, CL_MEM_READ_ONLY, sizeof(float)*sum_weight_elements(weights));
    input_buffers.push_back(cl::Buffer(c, CL_MEM_READ_ONLY, sizeof(float)*weights[0].size()));
    output_buffers.push_back(cl::Buffer(c, CL_MEM_READ_ONLY, sizeof(float)*weights[0].size()));
    for(size_t i = 0; i < weights.size(); ++i){
        input_buffers.push_back(cl::Buffer(c, CL_MEM_READ_WRITE, sizeof(float)*weights[i][0].size()));
        output_buffers.push_back(cl::Buffer(c, CL_MEM_READ_WRITE, sizeof(float)*weights[i][0].size()));
    }
}


Neurnet::Neurnet(std::vector<Layer> layers, float learnrate, KernelFunctor fp_ker, KernelFunctor bp_ker, std::vector<std::random_device::result_type> rs) :
learning_rate(learnrate),
randgen_seeds(rs),
weights(layers.size()-1, std::vector<std::vector<float>>(1, std::vector<float>(1,0))),
biases(layers.size(), std::vector<float>(1,0)),
n_layers(layers),
hit(0),
miss(0),
logfile("Log001.txt")
{
    std::function<float()> randgen = get_randgen(randgen_seeds);
    std::cout << "Generating weights" << std::endl;
    for(size_t z = 0; z < layers.size()-1; ++z){
        weights[z] = std::vector<std::vector<float>>(layers[z].n_number, std::vector<float>(layers[z+1].n_number,0));
        for(size_t y = 0; y <weights[z].size(); ++y){
            for(size_t x = 0; x < weights[z][y].size(); ++x){      //x is the current layer, y is the previous one
                weights[z][y][x] = randgen();
            }
        }
    }
    forprop_kernel = fp_ker;
    backprop_kernel = bp_ker;
    create_buffers(forprop_kernel.get_context());

    for(size_t i = 0; i < layers.size(); ++i){
        biases[i] = std::vector<float>(layers[i].n_number, 0);
        for(size_t j = 0; j < layers[i].n_number; ++j){
            biases[i][j] = randgen();
        }
    }
    std::cout << "Weights generated: Starting training" << std::endl;
}

Neurnet::~Neurnet()
{
    //dtor
}

std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>> Neurnet::forprop(std::vector<uint8_t> image){
    std::pair<std::vector<std::vector<float>>,std::vector<std::vector<float>>> ins_outs; //Pair of inputs and outputs for every neuron
    std::vector<std::vector<float>> inputs;
    std::vector<std::vector<float>> outputs;
    std::vector<float> temp(image.size(), 0);
    for(size_t i = 0; i < image.size(); ++i){
        temp[i] = image[i];
    }
    temp = temp/255; //Input normalization
    //temp += biases[0];
    //inputs.push_back(temp);
    forprop_kernel.c_queue.enqueueWriteBuffer(input_buffers[0], CL_FALSE, 0, sizeof(float)*temp.size(), &temp[0]);
    activate_choice(temp, n_layers[0].activator);
    outputs.push_back(temp);
    for(size_t i = 0; i < weights.size(); ++i){
        temp = matrix_mult(temp, weights[i]);
        //temp += biases[i+1];
        inputs.push_back(temp);
        activate_choice(temp, n_layers[i+1].activator);
        outputs.push_back(temp);
    }
    ins_outs = {inputs, outputs};
    return ins_outs;
}

std::vector<std::vector<float>> Neurnet::calc_deltas(std::vector<float> target, const std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>>& ins_outs){
    std::vector<std::vector<float>> deltas(ins_outs.second.size(), std::vector<float>(1,0));
    for(int i = ins_outs.second.size()-1; i >= 1; --i){ //Check for validity - may not need first layer deltas
        std::function<float(float)> derivative = derivative_choice(n_layers[i].activator);
        if(i == ins_outs.second.size()-1){
            std::vector<float> layer_deltas(ins_outs.second[i].size(), 0);
            for(size_t j = 0; j < ins_outs.second[i].size(); ++j){
                layer_deltas[j] = -(target[j]-ins_outs.second[i][j])*derive(derivative, ins_outs.second[i][j]); //Review
            }
            deltas[i] = layer_deltas;
        }else{
            std::vector<float> layer_deltas(ins_outs.second[i].size(), 0);
            for(size_t j = 0; j < weights[i].size(); ++j){
                float sumdelta = 0;
                for(size_t k = 0; k < weights[i][j].size(); ++k){
                    sumdelta += deltas[i+1][k]*weights[i][j][k];
                }
                layer_deltas[j] = sumdelta*derive(derivative, ins_outs.second[i][j]);
            }
            deltas[i] = layer_deltas;
        }
    }
    return deltas;
}

void Neurnet::backprop(std::vector<float> target, const std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>>& ins_outs, std::vector<std::vector<std::vector<float>>>& weights_update){
    std::vector<std::vector<float>> deltas = calc_deltas(target, ins_outs);
    for(size_t z = 0; z < weights.size(); ++z){
        for(size_t y = 0; y < weights[z].size(); ++y){
            for(size_t x = 0; x < weights[z][y].size(); ++x){
                weights_update[z][y][x] += learning_rate*ins_outs.first[z][y]*deltas[z+1][x];      ///REWRITE FOR MATRIXMATH
            }
        }
    }
    for(size_t i = 0; i < biases.size(); ++i){
        for(size_t j = 0; j < biases[i].size(); ++j){
            biases[i][j] -= learning_rate*deltas[i][j];
        }
    }
}

/**
* Checks if the output of a forward pass corresponds to the expected output.
* @param target The expected output
* @param actual The output of a forward pass
* @return True if the output matches the target, false otherwise
*/
bool out_check(uint8_t target, std::vector<float> actual){
    size_t top = 0;
    float maxi = 0;
    for(size_t i = 0; i < actual.size(); ++i){
        if(actual[i]>maxi){
            maxi = actual[i];
            top = i;
        }
    }
    if(target == top){
        return true;
    }else{
        return false;
    }
}

/**
* Calculates the squared total error of a forward pass.
* (The error is halved to simplify derivation)
* @param target The expected output
* @param actual The output of a forward pass
* @return The total error.
*/
float calc_total_error(uint8_t target, std::vector<float> actual){
    std::vector<float> t = gen_target(10, target);
    float err_tot = 0;
    for(size_t i = 0; i < actual.size(); ++i){
        err_tot += (pow(t[i]-actual[i],2))/2;
    }
    return err_tot;
}

void Neurnet::single_pass(uint8_t label, const std::vector<uint8_t>& image){
    std::vector<std::vector<float>> outputs = forprop(image).second;
    if(out_check(label, outputs.back())){
        ++hit;
    }else{
        ++miss;
    }
}

float Neurnet::train_pass(uint8_t label, std::vector<uint8_t> image, std::vector<std::vector<std::vector<float>>>& weights_update){
    const std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>>& ins_outs = forprop(image);
    std::vector<float> target = gen_target(10, label);
    backprop(target, ins_outs, weights_update);
    for(size_t i = 0; i < ins_outs.second.back().size(); ++i){
        logfile << ins_outs.second.back()[i] << ' ' << target[i] << ' ' << (pow(target[i]-ins_outs.second.back()[i],2))/2 << std::endl;
    }
    logfile << "--------" << std::endl;
    return calc_total_error(label, ins_outs.second.back());
}

void Neurnet::train_net(Dataset& training, int batchsize){
    std::vector<std::vector<std::vector<float>>> weights_update(weights);
    setvalue(weights_update,0);
    logfile << "------------Training error values------------" << std::endl;
    //int index = 0;
    float err_tot_sum = 0;
    while(training.check_over()){
        float err_tot = train_pass(training.get_label(), training.get_im(), weights_update);
        err_tot_sum += err_tot;
        /*if(training.get_index()%100 == 0){
            std::stringstream ss;
            ss << index;
            std::string fname = "w_mat" + ss.str() + ".txt";
            std::ofstream fout(fname);
            fout << weights;
            fout.close();
            ++index;
        }*/

        if(training.get_index()%batchsize == 0){
            weights -= weights_update/batchsize;
            setvalue(weights_update, 0);
            std::cout << training.get_index() << '/' << 60000 << std::endl;
            std::cout << "Total error:" << err_tot_sum/batchsize << std::endl;
            err_tot_sum = 0;
        }
        training.load_one();
    }
}

std::ostream& operator<<(std::ostream& out, std::vector<std::vector<std::vector<float>>> w){
    for(std::vector<std::vector<float>> vv : w){
        for(std::vector<float> v : vv){
            for(float d : v){
                out << d << ' ';
            }
            out << std::endl;
        }
        out << ';' << std::endl;
    }
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
    master << weights;
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
        single_pass(testing.get_label(), testing.get_im());
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

