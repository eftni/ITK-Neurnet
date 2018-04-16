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

/*std::vector<double> matrix_mult(std::vector<double> out, std::vector<std::vector<double>> weights){
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
}*/

std::vector<double> matrix_mult(std::vector<double> out, std::vector<std::vector<double>> weights){
    if(weights.size() != out.size()){
        std::cout << "MATRIXMULT ERROR: Matrix sizes don't match" << std::endl;
        exit(1);
    }
    std::vector<double> temp(weights[0].size(), 0);
    for(size_t outer = 0; outer < weights[0].size(); ++outer){
        for(size_t inner_mat = 0; inner_mat < weights.size(); ++inner_mat){  ///REVIEW INDEXING - URGENT!!!
            for(size_t inner_out = 0; inner_out < out.size(); ++inner_out){
                temp[outer] = weights[inner_mat][outer] * out[inner_out];
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

void setvalue(std::vector<std::vector<std::vector<double>>>& mat, double val){  ///PRIMITIVE - REPLACE ASAP
    for(std::vector<std::vector<double>>& vv : mat){
        for(std::vector<double>& v : vv){
            for(double& d : v){
                d = val;
            }
        }
    }
}

void operator-=(std::vector<std::vector<std::vector<double>>>& w, const std::vector<std::vector<std::vector<double>>>& update){ ///PRIMITIVE - REPLACE ASAP
    for(size_t i = 0; i < w.size(); ++i){
        for(size_t j = 0; j < w[i].size(); ++j){
            for(size_t k = 0; k < w[i][j].size(); ++k){
                w[i][j][k] -= update[i][j][k];
            }
        }
    }
}

std::vector<std::vector<std::vector<double>>> operator/(std::vector<std::vector<std::vector<double>>>& w, double d){ ///PRIMITIVE - REPLACE ASAP
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
