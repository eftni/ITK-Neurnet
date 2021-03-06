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
void warmup(std::function<double()>& randgen){
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
std::function<double()> get_randgen(std::vector<std::random_device::result_type>& seeds){
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
    std::function<double()> randgen1 = std::bind(std::uniform_real_distribution<double>(-1,1), std::mt19937(s));
    warmup(randgen1);
    return randgen1;
}


Neurnet::Neurnet(std::vector<Layer> layers, double learnrate, std::vector<std::random_device::result_type> rs) :
learning_rate(learnrate),
randgen_seeds(rs),
weights(layers.size()-1, std::vector<std::vector<double>>(1, std::vector<double>(1,0))),
biases(layers.size(), std::vector<double>(1,0)),
n_layers(layers),
hit(0),
miss(0),
logfile("Log001.txt")
{
    std::function<double()> randgen = get_randgen(randgen_seeds);
    std::cout << "Generating weights" << std::endl;
    for(size_t z = 0; z < layers.size()-1; ++z){
        weights[z] = std::vector<std::vector<double>>(layers[z].n_number, std::vector<double>(layers[z+1].n_number,0));
        for(size_t y = 0; y <weights[z].size(); ++y){
            for(size_t x = 0; x < weights[z][y].size(); ++x){      //x is the current layer, y is the previous one
                weights[z][y][x] = randgen();
            }
        }
    }
    for(size_t i = 0; i < layers.size(); ++i){
        biases[i] = std::vector<double>(layers[i].n_number, 0);
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

std::ostream& operator<< (std::ostream& out, std::vector<double> inputs){
    for(size_t i = 0; i < inputs.size(); ++i){
        out << inputs[i] << ' ' << tanh(inputs[i]) << std::endl;
    }
    return out;
}

std::ostream& operator<< (std::ostream& out, std::vector<std::vector<double>> outputs){
    for(size_t i = 0; i < outputs.size(); ++i){
        for(size_t j = 0; j < outputs[i].size(); ++j){
            out << outputs[i][j] << ' ';
        }
        out << std::endl;
    }
    return out;
}

std::vector<std::vector<double>> Neurnet::forprop(std::vector<std::vector<uint8_t>> image){
    std::vector<double> temp = mat_to_row(image);
    std::vector<std::vector<double>> outputs;
    temp = temp/255; //Input normalization
    //temp += biases[0];
    activate_choice(temp, n_layers[0].activator);
    outputs.push_back(temp);
    for(size_t i = 0; i < weights.size(); ++i){
        temp = matrix_mult(temp, weights[i]);
        //temp += biases[i+1];
        activate_choice(temp, n_layers[i+1].activator);
        outputs.push_back(temp);
    }
    return outputs;
}

std::vector<std::vector<double>> Neurnet::calc_deltas(std::vector<double> target, std::vector<std::vector<double>> outputs){
    std::vector<std::vector<double>> deltas(outputs.size(), std::vector<double>(1,0));
    for(int i = outputs.size()-1; i >= 1; --i){ //Check for validity - may not need first layer deltas
        std::function<double(double)> derivative = derivative_choice(n_layers[i].activator);
        if(i == outputs.size()-1){
            std::vector<double> layer_deltas(outputs[i].size(), 0);
            for(size_t j = 0; j < outputs[i].size(); ++j){
                layer_deltas[j] = -(target[j]-outputs[i][j])*derive(derivative, outputs[i-1][j]);
            }
            deltas[i] = layer_deltas;
        }else{
            std::vector<double> layer_deltas(outputs[i].size(), 0);
            for(size_t j = 0; j < weights[i].size(); ++j){
                double sumdelta = 0;
                for(size_t k = 0; k < weights[i][j].size(); ++k){
                    sumdelta += deltas[i+1][k]*weights[i][j][k];
                }
                layer_deltas[j] = sumdelta*derive(derivative, outputs[i-1][j]);
            }
            deltas[i] = layer_deltas;
        }
    }
    return deltas;
}

void Neurnet::backprop(std::vector<double> target, std::vector<std::vector<double>> outputs, std::vector<std::vector<std::vector<double>>>& weights_update){
    std::vector<std::vector<double>> deltas = calc_deltas(target, outputs);
    for(size_t z = 0; z < weights.size(); ++z){
        for(size_t y = 0; y < weights[z].size(); ++y){
            for(size_t x = 0; x < weights[z][y].size(); ++x){
                weights_update[z][y][x] += learning_rate*outputs[z][y]*deltas[z+1][x];      ///REWRITE FOR MATRIXMATH
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
bool out_check(uint8_t target, std::vector<double> actual){
    size_t top = 0;
    double maxi = 0;
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
double calc_total_error(uint8_t target, std::vector<double> actual){
    std::vector<double> t = gen_target(10, target);
    double err_tot = 0;
    for(size_t i = 0; i < actual.size(); ++i){
        err_tot += (pow(t[i]-actual[i],2))/2;
    }
    return err_tot;
}

void Neurnet::single_pass(uint8_t label, std::vector<std::vector<uint8_t>> image){
    std::vector<std::vector<double>> outputs = forprop(image);
    if(out_check(label, outputs.back())){
        ++hit;
    }else{
        ++miss;
    }
}

double Neurnet::train_pass(uint8_t label, std::vector<std::vector<uint8_t>> image, std::vector<std::vector<std::vector<double>>>& weights_update){
    std::vector<std::vector<double>> outputs = forprop(image);
    std::vector<double> target = gen_target(10, label);
    backprop(target, outputs, weights_update);
    for(size_t i = 0; i < outputs.back().size(); ++i){
        logfile << outputs.back()[i] << ' ' << target[i] << ' ' << (pow(target[i]-outputs.back()[i],2))/2 << std::endl;
    }
    logfile << "--------" << std::endl;
    return calc_total_error(label, outputs.back());
}

std::ostream& operator<<(std::ostream& out, std::vector<std::vector<std::vector<double>>> w){
    for(std::vector<std::vector<double>> vv : w){
        for(std::vector<double> v : vv){
            for(double d : v){
                out << d << ' ';
            }
            out << std::endl;
        }
        out << ';' << std::endl;
    }
    return out;
}

void Neurnet::train_net(Dataset& training, int batchsize){
    std::vector<std::vector<std::vector<double>>> weights_update(weights);
    setvalue(weights_update,0);
    logfile << "------------Training error values------------" << std::endl;
    int index = 0;
    double err_tot_sum = 0;
    while(training.check_over()){
        double err_tot = train_pass(training.get_label(), training.get_im(), weights_update);
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

void Neurnet::write_to_master(){
    std::ofstream master("master.txt");
    master << "-------Current best network:-------" << std::endl;
    master << "Percentage: " << ((double)hit/10000)*100 << std::endl;
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
double read_master_best(){
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
    std::cout << "ratio: " << ((double)hit/10000)*100 << '%' << std::endl;
    double best = read_master_best();
    if(((double)hit/10000)*100 > best){
        write_to_master();
    }
    logfile << "hit: " << hit << " miss: " << miss << std::endl;
    logfile << "ratio: " << ((double)hit/10000)*100 << '%' << std::endl;
    logfile.close();
}

