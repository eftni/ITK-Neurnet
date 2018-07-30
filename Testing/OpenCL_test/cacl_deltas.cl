float sigmoid(float x){
	return 1/(1+exp(-x));
}

float derive(float x, const unsigned int act_func){
	switch(act_func){
		case 0: return 1;
		case 1: return sigmoid(x)*(1-sigmoid(x));
		case 2: return 1-pow(tanh(x), 2);
	}
}

void kernel calc_deltas(const global float* inputs, const global float* outputs, const global float* target, const unsigned int act_func, global float* deltas_next){
	const int row_count = get_global_id(0);
	deltas_next[row_count] = -(target[row_count]-outputs[row_count])*derive(inputs[row_count], act_func);
}

void kernel calc_deltas(const global float* inputs, const global float* outputs, const global float* weights, const int wsize, const unsigned int act_func, global float* deltas_prev, global float* deltas_next){
	const int row_count = get_global_id(0);
	float acc = 0.0f;
	for(int i = 0; i < wsize, ++i){
		acc += deltas_prev[i]*weights[row_count*wsize+i]; /*Rowcount is previous layer or number of neurons, wsize is number of next layer neurons*/
	}
	deltas[row_count] = acc*derive(inputs[rowcount], act_func);
}


