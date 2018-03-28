#include "Neurnet.h"
#include "random"
#include "iostream"
#include "functional"
#include "..\\matrixmath.h"
#include "time.h"
#include "stdio.h"
#include "math.h"

void warmup(std::function<double()>& randgen){
    for(int i = 0; i <= 100000; ++i){
        randgen();
    }
}

std::function<double()> get_randgen(std::vector<std::random_device::result_type>& seeds){
    std::cout << "Creating random generator" << std::endl;
    srand(time(nullptr));
    //std::random_device r;
    for(int i = 0; i <= 7; ++i){    ///RANDOM DEVICE IMPLEMENTED INCORRECTLY: USE BOOST
        //seeds[i] = r();
        seeds[i] = rand();
        //std::cout << seeds[i] << std::endl;
    }
    std::seed_seq s{seeds[0],seeds[1],seeds[2],seeds[3],seeds[4],seeds[5],seeds[6],seeds[7]};
    std::function<double()> randgen1 = std::bind(std::uniform_real_distribution<double>(-1,1), std::mt19937(s));
    warmup(randgen1);
    return randgen1;
}


Neurnet::Neurnet(std::vector<int> layer_count, double learnrate, std::function<double(double)> activator, std::function<double(double)> derivative) :
learning_rate(learnrate),
act_func(activator),
act_func_derivative(derivative),
randgen_seeds(8, 0),
weights(layer_count.size()-1, std::vector<std::vector<double>>(1, std::vector<double>(1,0))),
logfile("Log001.txt")
{
    std::function<double()> randgen = get_randgen(randgen_seeds);
    std::cout << "Generating weights" << std::endl;
    for(size_t z = 0; z < layer_count.size()-1; ++z){
        //weights[z] = std::vector<std::vector<double>>(layer_count[z+1], std::vector<double>(layer_count[z],0));
        weights[z] = std::vector<std::vector<double>>(layer_count[z], std::vector<double>(layer_count[z+1],0));
        for(size_t y = 0; y <weights[z].size(); ++y){
            for(size_t x = 0; x < weights[z][y].size(); ++x){      ///x is the current layer, y is the previous one
                weights[z][y][x] = randgen();
                //std::cout << randgen() << std::endl;
            }
        }
    }
    std::cout << "Weights generated: Starting training" << std::endl;
}

Neurnet::~Neurnet()
{
    //dtor
}


/*template<typename T>
std::ostream& operator<<(std::ostream& out, std::vector<T> v){
    for(T t : v){
        out << t << ' ';
    }
    out << std::endl;
    return out;
}*/

std::vector<std::vector<double>> Neurnet::forprop(std::vector<std::vector<uint8_t>> image){
    std::vector<double> temp = mat_to_row(image);
    std::vector<std::vector<double>> outputs;
    activate(temp, act_func);
    outputs.push_back(temp);
    for(std::vector<std::vector<double>> w_set : weights){
        temp = matrix_mult(temp, w_set);
        activate(temp, act_func);
        outputs.push_back(temp);
    }
    return outputs;
}

std::vector<std::vector<double>> Neurnet::calc_deltas(std::vector<double> target, std::vector<std::vector<double>> outputs){
    std::vector<std::vector<double>> deltas(outputs.size(), std::vector<double>(1,0));
    for(int i = outputs.size()-1; i >= 0; --i){
        if(i == outputs.size()-1){
            std::vector<double> layer_deltas(outputs[i].size(), 0);
            for(size_t j = 0; j < outputs[i].size(); ++j){     ///WATCH THE NEGATIVE SIGN
                layer_deltas[j] = -(target[j]-outputs[i][j])*act_func_derivative(outputs[i][j]);
            }
            deltas[i] = layer_deltas;
        }else{
            std::vector<double> layer_deltas(outputs[i].size(), 0);
            for(size_t j = 0; j < weights[i].size(); ++j){
                double sumdelta = 0;
                for(size_t k = 0; k < weights[i][j].size(); ++k){
                    sumdelta += deltas[i+1][k]*weights[i][j][k];    ///--------CHECK INDEXING - DELTAS NaN - FIXED?
                }
                layer_deltas[j] = sumdelta*act_func_derivative(outputs[i][j]);
            }
            deltas[i] = layer_deltas;
        }
    }
    return deltas;
}

void Neurnet::backprop(std::vector<double> target, std::vector<std::vector<double>> outputs){
    //double total_error = calc_error(target, output);
    std::vector<std::vector<std::vector<double>>> weights_update(weights);
    std::vector<std::vector<double>> deltas = calc_deltas(target, outputs);
    for(size_t z = 0; z < weights.size(); ++z){
        for(size_t y = 0; y < weights[z].size(); ++y){
            for(size_t x = 0; x < weights[z][y].size(); ++x){
                weights_update[z][y][x] -= learning_rate*outputs[z][y]*deltas[z+1][x];
            }
        }
    }
    weights = weights_update;
}

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

/*std::ostream& operator<<(std::ostream& out, std::vector<std::vector<double>> mat){
    for(std::vector<double> v : mat){
        for(double d : v){
            std::cout << d << ' ';      ///DEBUG FUNCTION
        }
        std::cout << std::endl;
    }
    return out;
}*/

double Neurnet::train_pass(uint8_t label, std::vector<std::vector<uint8_t>> image){
    std::vector<std::vector<double>> outputs = forprop(image);
    backprop(gen_target(10, label), outputs);
    return calc_total_error(label, outputs.back());
}

void Neurnet::train_net(Dataset& training){
    logfile << "------------Training error values------------" << std::endl;
    while(training.check_over()){
        double err_tot = train_pass(training.get_label(), training.get_im());
        if(training.get_index()%500 == 0){
            std::cout << training.get_index() << '/' << 60000 << std::endl;
            std::cout << "Total error:" << err_tot << std::endl;
        }
        logfile << err_tot << std::endl;
        ///MINIBATCH
        training.load_one();
    }
}

void Neurnet::test_net(Dataset& testing){
    logfile << "------------Testing hit rate------------" << std::endl;
    while(testing.check_over()){
        train_pass(testing.get_label(), testing.get_im());
        if(testing.get_index()%500 == 0){
            std::cout << testing.get_index() << '/' << 10000 << std::endl;
        }
        testing.load_one();
    }
    logfile << "hit: " << hit << " miss: " << miss << std::endl;
    logfile << "ratio: " << (double)hit/(double) miss << std::endl;
    logfile.close();
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
