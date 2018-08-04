#ifndef GPU_MATRIXMATH_H_INCLUDED
#define GPU_MATRIXMATH_H_INCLUDED
#include "KernelFunctor.h"




std::vector<std::vector<float>> calc_deltas(Neurnet& network, std::vector<float> target, const std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>>& ins_outs){
    std::vector<std::vector<float>> deltas(ins_outs.second.size(), std::vector<float>(1,0));
    for(int i = ins_outs.second.size()-1; i >= 1; --i){ //Check for validity - may not need first layer deltas
        std::function<float(float)> derivative = derivative_choice(network.n_layers[i].activator);
        if(i == ins_outs.second.size()-1){
            std::vector<float> layer_deltas(ins_outs.second[i].size(), 0);
            for(size_t j = 0; j < ins_outs.second[i].size(); ++j){
                layer_deltas[j] = -(target[j]-ins_outs.second[i][j])*derive(derivative, ins_outs.second[i][j]); //Review
            }
            deltas[i] = layer_deltas;
        }else{
            std::vector<float> layer_deltas(ins_outs.second[i].size(), 0);
            for(size_t j = 0; j < network.weights[i].size(); ++j){
                float sumdelta = 0;
                for(size_t k = 0; k < network.weights[i][j].size(); ++k){
                    sumdelta += deltas[i+1][k]*network.weights[i][j][k];
                }
                layer_deltas[j] = sumdelta*derive(derivative, ins_outs.second[i][j]);
            }
            deltas[i] = layer_deltas;
        }
    }
    return deltas;
}

void backprop(Neurnet& network, std::vector<float> target, const std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>>& ins_outs, std::vector<std::vector<std::vector<float>>>& weights_update){
    std::vector<std::vector<float>> deltas = calc_deltas(network, target, ins_outs);
    for(size_t z = 0; z < network.weights.size(); ++z){
        for(size_t y = 0; y < network.weights[z].size(); ++y){
            for(size_t x = 0; x < network.weights[z][y].size(); ++x){
                weights_update[z][y][x] += network.learning_rate*ins_outs.first[z][y]*deltas[z+1][x];      ///REWRITE FOR MATRIXMATH
            }
        }
    }
    /*for(size_t i = 0; i < biases.size(); ++i){
        for(size_t j = 0; j < biases[i].size(); ++j){
            biases[i][j] -= learning_rate*deltas[i][j];
        }
    }*/
}

#endif // GPU_MATRIXMATH_H_INCLUDED
