float sigmoid(float x){
	return 1/(1+exp(-x));
}

float derive(float x, const unsigned int act_func){
	switch(act_func){
		case 0: return x;
		case 1: return sigmoid(x)*(1-sigmoid(x));
		case 2: return 1-pow(tanh(x), 2);
	}
	return 0;
}

void kernel calc_deltas(int first, const global float* inputs, const global float* outputs, const global float* weights, const int wsize, const global char* targets,
						const unsigned int act_func, const global float* deltas_prev, const int deltas_prev_size, global float* deltas_next){

	const int output_size = get_global_size(0);
	const int output_index = get_global_id(0);
	const int batch_element = get_global_id(1);
	if(first == 0){
		if(targets[batch_element] == output_index){
			deltas_next[batch_element*output_size + output_index] = -(1-outputs[batch_element*output_size + output_index])*derive(outputs[batch_element*output_size + output_index], act_func);
			deltas_next[batch_element*output_size + output_index] = 500;
		}else{
			deltas_next[batch_element*output_size + output_index] = -(0-outputs[batch_element*output_size + output_index])*derive(outputs[batch_element*output_size + output_index], act_func);
			deltas_next[batch_element*output_size + output_index] = 600;
		}
		
	}else{
		float acc = 0.0f;
		for(int i = 0; i < deltas_prev_size; ++i){
			acc += deltas_prev[batch_element*deltas_prev_size + i]*weights[i*wsize+output_index];
		}
		deltas_next[batch_element*output_size + output_index] = acc*derive(outputs[batch_element*output_size + output_index], act_func);
		deltas_next[batch_element*output_size + output_index] = 700;
	}
}


