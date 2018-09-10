void kernel backprop(const global float* deltas, const global float* outputs, global float* w_update, const int wsize){
	const int row_count = get_global_id(0);
	const int col_count = get_global_id(1);
	w_update[row_count*wsize+col_count] = outputs[row_count] * deltas[col_count];
}