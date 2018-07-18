void kernel matrix_mult(global const float* input, global const float* w, const int len, global float* output){
	const int row_count = get_global_id(0);
    for(int i = 0; i < len; ++i){
		output[row_count] += input[i] * w[row_count*len + i];
	}
}