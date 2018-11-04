void kernel backprop(const global float* deltas, const global float* outputs, const int neurons, global float* w_update, const int wsize, const float learn_rate){
	const int out_index = get_global_id(0);
	const int delta_index = get_global_id(1);
	const int batch_element = get_global_id(2);
	w_update[delta_index*neurons+out_index] = outputs[batch_element*neurons + out_index] * deltas[batch_element*wsize + delta_index] * learn_rate;
	printf("%s%f%s%f%s%f\n", "Outputs: ", outputs[batch_element*neurons + out_index], " Deltas: ", deltas[batch_element*wsize + delta_index], " Result: ", outputs[batch_element*neurons + out_index] * deltas[batch_element*wsize + delta_index]);
	printf("%s%f\n", "Learning rate: ", learn_rate);
}