
float kernel identity(float x){
	return x;
}

float kernel sigmoid(float x){
	return x;
}

float kernel hyp_tan(float x){
	return x;
}


void kernel forprop(global const float* input, global const float* w, const unsigned int wsize, global float* output, const unsigned int act_func, global float* act_output){
   const int row_count = get_global_id(0);
   const int col_count = get_global_id(1);
   float acc = 0.0f;
   for(int i = 0; i < wsize; ++i){
       acc += input[i*msize+row_count] * w[col_count*wsize + i];
   }
   output[col_count*msize+row_count] = acc;
};