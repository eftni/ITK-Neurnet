void kernel backprop(const global float* deltas, const global float* inputs, global float* w_update, const int wsize){
	const int row_count = get_global_id(0);
	const int col_count = get_global_id(1);
	w_update[row_count*wsize+col_count] = inputs[row_count] * deltas[col_count]; //Inputs is 16, deltas is 10
}