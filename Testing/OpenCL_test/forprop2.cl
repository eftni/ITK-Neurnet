float sigmoid(float x){
	return 1/(1+exp(-x));
}

float activate(float x, const unsigned int act_func){
	switch(act_func){
		case 0: return x;
		case 1: return sigmoid(x);
		case 2: return tanh(x);
	}
	return 0;
}


void kernel forprop(global const float* input, global const float* w, const unsigned int tilesize,  const unsigned int wsize, global float* output, const unsigned int act_func, global float* act_output){
   const int w_index = get_global_id(0);
   const int tile = get_local_id(0);
   const int output_index = get_global_id(1);
   for(int i = 0; i < tilesize; ++i){
       output[output_index] += input[tile*tilesize + i] * w[output_index*wsize + tile*tilesize + i];
   }
   barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
   act_output[output_index] = activate(output[output_index], act_func);
};