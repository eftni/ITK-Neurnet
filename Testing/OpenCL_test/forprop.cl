float sigmoid(float x){
	return 1/(1+exp(-x));
}

float activate(float x, const unsigned int act_func){
	switch(act_func){
		case 0: return x;
		case 1: return sigmoid(x);
		case 2: return tanh(x);
	}
}


void kernel forprop(global const float* input, global const float* w, const unsigned int wsize, global float* output, const unsigned int act_func, global float* act_output){
   const int row_count = get_global_id(0);
   float acc = 0.0f;
   for(int i = 0; i < wsize; ++i){
       acc += input[i] * w[row_count*wsize + i];
   }
   output[row_count] = acc;
   act_output[row_count] = activate(acc, act_func);
};