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



void kernel forprop(global const float* input, const int input_size, global const float* w, const unsigned int wsize, global float* output, const int output_size, const unsigned int act_func, global float* act_output){
	const int output_index = get_global_id(0);
	const int batch_element = get_global_id(1);
	float acc = 0.0f;
	for(int i = 0; i < wsize; ++i){
		acc += input[batch_element*input_size + i] * w[output_index*wsize + i];
	}
	output[batch_element*output_size + output_index] = acc;
	act_output[batch_element*output_size + output_index] = activate(acc, act_func);
};