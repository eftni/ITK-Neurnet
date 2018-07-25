
void kernel matrix_mult(global const float* input, global const float* m, const int msize, global float* output){
   const int row_count = get_global_id(0);
   const int col_count = get_global_id(1);
   float acc = 0.0f;
   for(int i = 0; i < msize; ++i){
       acc += input[i*msize+row_count] * m[col_count*msize + i];
   }
   output[col_count*msize+row_count] = acc;
};