#ifndef MATRIXMATH_H_INCLUDED
#define MATRIXMATH_H_INCLUDED
#include "math.h"

std::vector<double> mat_to_row(std::vector<std::vector<uint8_t>> mat){
    std::vector<double> temp(mat.size()*mat[0].size(),0);
    for(size_t i = 0; i < mat.size(); ++i){
        for(size_t j = 0; j < mat[i].size(); ++j){
            temp[(i*mat.size() + j)] = mat[i][j];
        }
    }
    return temp;
}

std::vector<double> matrix_mult(std::vector<double> out, std::vector<std::vector<double>> weights){
    if(weights[0].size() != out.size()){
        std::cout << "MATRIXMULT ERROR: Matrix sizes don't match" << std::endl;
        exit(1);
    }
    std::vector<double> temp(weights.size(), 0);
    for(size_t outer = 0; outer < weights.size(); ++outer){
        for(size_t inner_mat = 0; inner_mat < weights[outer].size(); ++inner_mat){
            for(size_t inner_out = 0; inner_out < out.size(); ++inner_out){
                temp[outer] = weights[outer][inner_mat] * out[inner_out];
            }
        }
    }
    return temp;
}

inline void activate(std::vector<double>& input, std::function<double(double)> activator){
    for(double& d : input){
        d = activator(d);
    }
}

std::vector<double> gen_target(int vect_size, int label){
    std::vector<double> temp(vect_size, 0);
    temp[label] = 1;        ///ADJUST FOR ACTIVATOR FUNCTION
    return temp;
}

/*double calc_error(std::vector<double> target, std::vector<double> output){
    if(target.size() != output.size()){
        std::cout << "E-SIGNAL CALCULATION ERROR: target and output sizes don't match";
        exit(1);
    }
    double total_error = 0;
    for(int i = 0; i < target.size(); ++i){
        total_error += pow(target[i]-output[i] ,2)/2;
    }
    return total_error;
}*/


#endif // MATRIXMATH_H_INCLUDED
