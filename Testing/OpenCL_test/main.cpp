#include <iostream>
#include "include//KernelFunctor.h"
#include "vector"

using namespace std;

void operator/(std::vector<float>& v, float d){
    for(float& f : v){
        f = f/d;
    }
}

int main()
{
    KernelFunctor forprop_kernel("forprop.cl");
    std::vector<float> input = {1,2,3,4,5};
    input/255;
    std::vector<float> w = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    std::vector<float> output(3, 0);
    cl::Buffer input_buffer = cl::Buffer(forprop_kernel.get_context(), CL_MEM_READ_WRITE, sizeof(float)*input.size());
    cl::Buffer w_buffer = cl::Buffer(forprop_kernel.get_context(), CL_MEM_READ_WRITE, sizeof(float)*w.size());
    cl::Buffer output_buffer = cl::Buffer(forprop_kernel.get_context(), CL_MEM_READ_WRITE, sizeof(float)*output.size());
    cl::Buffer act_output_buffer = cl::Buffer(forprop_kernel.get_context(), CL_MEM_READ_WRITE, sizeof(float)*output.size());
    forprop_kernel.c_queue.enqueueWriteBuffer(input_buffer, CL_FALSE, 0, sizeof(float)*input.size(), &input[0]);
    forprop_kernel.c_queue.enqueueWriteBuffer(w_buffer, CL_FALSE, 0, sizeof(float)*w.size(), &w[0]);
    forprop_kernel(cl::NullRange, cl::NDRange(3), cl::NullRange, input_buffer, w_buffer, 5, output_buffer, 2, act_output_buffer);
    forprop_kernel.c_queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0, sizeof(float)*output.size(), &output[0]);
    for(float f : output){
        std::cout << f << std::endl;
    }
    std::cout << std::endl;
    forprop_kernel.c_queue.enqueueReadBuffer(act_output_buffer, CL_TRUE, 0, sizeof(float)*output.size(), &output[0]);
    for(float f : output){
        std::cout << f << std::endl;
    }
    std::cout << std::endl;
    KernelFunctor delta_kernel("calc_deltas.cl");
    cl::Buffer delta_buffer = cl::Buffer(delta_kernel.get_context(), CL_MEM_READ_WRITE, sizeof(float)*output.size());
    cl::Buffer next_delta_buffer = cl::Buffer(delta_kernel.get_context(), CL_MEM_READ_WRITE, sizeof(float)*input.size());
    cl::Buffer target_buffer = cl::Buffer(delta_kernel.get_context(), CL_MEM_READ_WRITE, sizeof(float)*output.size());
    std::vector<float> target = {0,0,1};
    delta_kernel.c_queue.enqueueWriteBuffer(target_buffer, CL_TRUE, 0, sizeof(float)*target.size(), &target[0]);
    cl::Buffer dummy = cl::Buffer(delta_kernel.get_context(), CL_MEM_READ_WRITE, 0);
    delta_kernel(cl::NullRange, cl::NDRange(3), cl::NullRange, 0, output_buffer, act_output_buffer, w_buffer, 3, target_buffer, 0, dummy, delta_buffer);
    std::vector<float> deltas(3,0);
    delta_kernel.c_queue.enqueueReadBuffer(delta_buffer, CL_TRUE, 0, sizeof(float)*deltas.size(), &deltas[0]);
    for(float f : deltas){
        std::cout << f << std::endl;
    }
    std::cout << std::endl;
    delta_kernel(cl::NullRange, cl::NDRange(5), cl::NullRange, 1, input_buffer, output_buffer, w_buffer, 3, target_buffer, 0, delta_buffer, next_delta_buffer);
    std::vector<float> next_deltas(5,0);
    delta_kernel.c_queue.enqueueReadBuffer(next_delta_buffer, CL_TRUE, 0, sizeof(float)*next_deltas.size(), &next_deltas[0]);
    for(float f : next_deltas){
        std::cout << f << std::endl;
    }
    std::cout << std::endl;
    KernelFunctor backprop_kernel("backprop.cl");
    cl::Buffer w_update_buffer = cl::Buffer(backprop_kernel.get_context(), CL_MEM_READ_WRITE, sizeof(float)*w.size());
    backprop_kernel(cl::NullRange, cl::NDRange(5, 3), cl::NullRange, delta_buffer, input_buffer, w_update_buffer, 3);
    std::vector<float> w_update(15,0);
    backprop_kernel.c_queue.enqueueReadBuffer(w_update_buffer, CL_TRUE, 0, sizeof(float)*w.size(), &w_update[0]);
    for(float f : w_update){
        std::cout << f << ' ';
    }
    std::cout << std::endl;
    return 0;
}
