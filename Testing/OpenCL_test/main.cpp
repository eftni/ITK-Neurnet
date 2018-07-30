#include <iostream>
#include "KernelFunctor.h"
#include "vector"

using namespace std;

int main()
{
    KernelFunctor forprop_kernel("forprop.cl");
    std::vector<float> input = {1,2,3,4,5};
    std::vector<float> w = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    std::vector<float> output(3, 0);
    std::vector<float> test(5, 0);
    cl::Buffer input_buffer = cl::Buffer(forprop_kernel.get_context(), CL_MEM_READ_WRITE, sizeof(float)*input.size());
    cl::Buffer w_buffer = cl::Buffer(forprop_kernel.get_context(), CL_MEM_READ_WRITE, sizeof(float)*w.size());
    cl::Buffer output_buffer = cl::Buffer(forprop_kernel.get_context(), CL_MEM_READ_WRITE, sizeof(float)*output.size());
    cl::Buffer act_output_buffer = cl::Buffer(forprop_kernel.get_context(), CL_MEM_READ_WRITE, sizeof(float)*output.size());
    forprop_kernel.c_queue.enqueueWriteBuffer(input_buffer, CL_FALSE, 0, sizeof(float)*input.size(), &input[0]);
    forprop_kernel.c_queue.enqueueWriteBuffer(w_buffer, CL_FALSE, 0, sizeof(float)*w.size(), &w[0]);
    forprop_kernel(cl::NullRange, cl::NDRange(3), cl::NullRange, input_buffer, w_buffer, 5, output_buffer, 0, act_output_buffer);
    forprop_kernel.c_queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0, sizeof(float)*output.size(), &output[0]);
    for(float f : output){
        std::cout << f << std::endl;
    }
    std::cout << std::endl;
    forprop_kernel.c_queue.enqueueReadBuffer(act_output_buffer, CL_TRUE, 0, sizeof(float)*output.size(), &output[0]);
    for(float f : output){
        std::cout << f << std::endl;
    }
    return 0;
}
