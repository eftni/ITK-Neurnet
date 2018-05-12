#ifndef MATRIXMATH_H_INCLUDED
#define MATRIXMATH_H_INCLUDED
#include "math.h"

/**
* Convert a two dimensional array to a one dimensional vector
* @param mat An image in matrix (2D) form
* @return The image in 1D vector form
*/
std::vector<double> mat_to_row(std::vector<std::vector<uint8_t>> mat){
    std::vector<double> temp(mat.size()*mat[0].size(),0);
    for(size_t i = 0; i < mat.size(); ++i){
        for(size_t j = 0; j < mat[i].size(); ++j){
            temp[(i*mat.size() + j)] = mat[i][j];
        }
    }
    return temp;
}

/**
* Multiplies the outputs of a layer by the weights going to the next layer
* @param out The outputs of a single layer
* @param weights The weight matrix between the two layers
* @return The product of the two matrices
*/
std::vector<double> matrix_mult(std::vector<double> out, std::vector<std::vector<double>> weights){
    if(weights.size() != out.size()){
        std::cout << "MATRIXMULT ERROR: Matrix sizes don't match" << std::endl;
        exit(1);
    }
    std::vector<double> result(weights[0].size(), 0);
    for(size_t i = 0; i < result.size(); ++i){
        for(size_t j = 0; j < out.size(); ++j){
            result[i] += weights[j][i] * out[j];
        }
    }
    return result;
}

/**
* Takes the inputs of a given layer and passes them to that layer's activation function
* @param input Vector of neuron inputs
* @param activator The activation function to be used
*/
inline void activate(std::vector<double>& input, std::function<double(double)> activator){
    for(double& d : input){
        d = activator(d);
    }
}

/**
* Generates a target vector based on and image label.
* The expected output on the neuron corresponding to the correct output depends on the
* activation function used, but is usually 1. Likewise, for every other neuron, the expected
* output may vary, but is usually 0.
* @param vect_size The size of the output layer
* @param label The expected output
* @return A vector matching the output layer in size
*/
std::vector<double> gen_target(int vect_size, int label){
    std::vector<double> temp(vect_size, 0);
    temp[label] = 1;        //ADJUST FOR ACTIVATOR FUNCTION
    return temp;
}

void setvalue(std::vector<std::vector<std::vector<double>>>& mat, double val){  //PRIMITIVE - REPLACE ASAP
    for(std::vector<std::vector<double>>& vv : mat){
        for(std::vector<double>& v : vv){
            for(double& d : v){
                d = val;
            }
        }
    }
}

void operator+=(std::vector<double>& outputs, const std::vector<double>& biases){ //PRIMITIVE - REPLACE ASAP
    for(size_t i = 0; i < outputs.size(); ++i){
        outputs[i] += biases[i];
    }
}

void operator-=(std::vector<std::vector<std::vector<double>>>& w, const std::vector<std::vector<std::vector<double>>>& update){ //PRIMITIVE - REPLACE ASAP
    for(size_t i = 0; i < w.size(); ++i){
        for(size_t j = 0; j < w[i].size(); ++j){
            for(size_t k = 0; k < w[i][j].size(); ++k){
                w[i][j][k] -= update[i][j][k];
            }
        }
    }
}

std::vector<double> operator/(std::vector<double> v, double d){ //PRIMITIVE - REPLACE ASAP
    for(size_t i = 0; i < v.size(); ++i){
        v[i] = v[i]/d;
    }
    return v;
}

std::vector<std::vector<std::vector<double>>> operator/(std::vector<std::vector<std::vector<double>>>& w, double d){ //PRIMITIVE - REPLACE ASAP
    for(size_t i = 0; i < w.size(); ++i){
        for(size_t j = 0; j < w[i].size(); ++j){
            for(size_t k = 0; k < w[i][j].size(); ++k){
                w[i][j][k] = w[i][j][k]/d;
            }
        }
    }
    return w;
}

void set_pair_field(std::vector<std::pair<double, double>> v_p, std::vector<double> val, bool output){
    if(v_p.size() != val.size()){
        std::cerr << "Matrix sizes don't match" << std::endl;
        exit(1);
    }else{
        if(!output){
            for(size_t i = 0; i < v_p.size(); ++i){
                v_p[i].first = val[i];
            }
        }else{
            for(size_t i = 0; i < v_p.size(); ++i){
                v_p[i].first = val[i];
            }
        }
    }
}

#endif // MATRIXMATH_H_INCLUDED
