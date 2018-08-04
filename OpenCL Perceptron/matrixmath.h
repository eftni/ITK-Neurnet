#ifndef MATRIXMATH_H_INCLUDED
#define MATRIXMATH_H_INCLUDED
#include "math.h"

/**
* Convert a two dimensional array to a one dimensional vector
* @param mat An image in matrix (2D) form
* @return The image in 1D vector form
*/
std::vector<float> mat_to_row(std::vector<std::vector<uint8_t>> mat){ //Modify file-reader
    std::vector<float> temp(mat.size()*mat[0].size(),0);
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
std::vector<float> matrix_mult(std::vector<float> out, std::vector<std::vector<float>> weights){
    if(weights.size() != out.size()){
        std::cout << "MATRIXMULT ERROR: Matrix sizes don't match" << std::endl;
        exit(1);
    }
    std::vector<float> result(weights[0].size(), 0);
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
template<typename func>
inline void activate(std::vector<float>& input, func activator){
    for(float& d : input){
        d = activator(d);
    }
}

/**
* Selects the activator function to be used for a given layer
* @param input Vector of neuron inputs
* @param activator The activation function to be used
*/
void activate_choice(std::vector<float>& input, act_func_type activator){
    switch (activator){
        case identity: activate(input, [](float x){return x;});
        break;
        case hyp_tan: activate(input, tanh);
        break;
        case sigmoid: activate(input, [](float x){return 1/(1-exp(-x));});
        break;
    }
}

template<typename func>
inline float derive(func derivative, float x){
    return derivative(x);
}

std::function<float(float)> derivative_choice(act_func_type activator){
    switch (activator){
        case identity: return [](float x){return 1;};
        case hyp_tan: return [](float x){return 1-pow(tanh(x),2);};
        case sigmoid: return [](float x){return (1/(1-exp(-x)))*(1-(1/(1-exp(-x))));};
    }
    return [](float x){return 1;};
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
std::vector<float> gen_target(int vect_size, int label){
    std::vector<float> temp(vect_size, 0);
    temp[label] = 1;        //ADJUST FOR ACTIVATOR FUNCTION
    return temp;
}

void setvalue(std::vector<float>& mat, float val){
    for(size_t i = 0; i < mat.size(); ++i){
        mat[i] = val;
    }
}

template<typename T>
void setvalue(std::vector<T>& mat, float val){
    for(size_t i = 0; i < mat.size(); ++i){
        setvalue(mat[i], val);
    }
}


void operator+=(std::vector<float>& outputs, const std::vector<float>& biases){
    for(size_t i = 0; i < outputs.size(); ++i){
        outputs[i] += biases[i];
    }
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

void set_pair_field(std::vector<std::pair<float, float>> v_p, std::vector<float> val, bool output){
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
