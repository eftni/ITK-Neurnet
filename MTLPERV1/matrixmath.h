#ifndef MATRIXMATH_H_INCLUDED
#define MATRIXMATH_H_INCLUDED

std::vector<double> mat_to_row(std::vector<std::vector<double>> mat){
    std::vector<double> temp(mat.size()*mat[0].size(),0);
    for(int i = 0; i < mat.size(); ++i){
        for(imt j = 0; j < mat[i].size(); ++j){
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
    for(int outer = 0; outer < weights.size(); ++outer){
        for(int inner_mat = 0; inner_mat < weights[outer].size(); ++inner_mat){
            for(int inner_out = 0; inner_out < out.size(); ++inner_out){
                temp[outer] = weights[outer][inner_mat] * out[inner_out];
            }
        }
    }
    return temp;
}


#endif // MATRIXMATH_H_INCLUDED
