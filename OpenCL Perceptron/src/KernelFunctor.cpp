#include "KernelFunctor.h"
#include "vector"
#include "iostream"
#include "stdlib.h"
#include "fstream"
#include "cstdarg"

std::string KernelFunctor::fetch_kernel_code(std::string fname){
    std::ifstream fin(fname);
    if(fin.good()){
        return std::string((std::istreambuf_iterator<char>(fin)), std::istreambuf_iterator<char>());
    }else{
        std::cout << fname << ": No such file found!" << std::endl;
        exit(4);
    }
}

KernelFunctor::KernelFunctor()
{

}

KernelFunctor::KernelFunctor(std::string fname)
{
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if(all_platforms.size() == 0){
        std::cout << "Error: No platforms available!" << std::endl;
        exit(2);
    }
    def_platform = all_platforms[0];
    std::vector<cl::Device> all_GPUs;
    def_platform.getDevices(CL_DEVICE_TYPE_GPU, &all_GPUs);
    if(all_GPUs.size() == 0){
        std::cout << "Error: No GPU found!" << std::endl;
        exit(3);
    }
    def_device = all_GPUs[0];
    def_device_context = cl::Context({def_device});
    std::string temp_code = fetch_kernel_code(fname);
    kernel_code.push_back({temp_code.c_str(), temp_code.length()});
    kernel_program = cl::Program({def_device_context, kernel_code});
    if(kernel_program.build({def_device}) != CL_SUCCESS){
        std::cout << "BUILD ERROR: " << kernel_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(def_device) << std::endl;
        exit(6);
    }
    c_queue = cl::CommandQueue({def_device_context, def_device});
    def_kernel = cl::Kernel(kernel_code, fname.substr(0, fname.length()-2).c_str());
}

KernelFunctor::~KernelFunctor()
{
    //dtor
}

template<typename T>
void KernelFunctor::set_argument(int num, T arg){
    def_kernel.setArg(num, arg);
}

template<typename T, typename... Targs>
void KernelFunctor::set_argument(int num, T first, Targs... args){
    def_kernel.setArg(num, first);
    set_argument(num+1, args...);
}

template<typename... Targs>
void KernelFunctor::operator()(cl::NDRange offset, cl::NDRange threads, cl::NDRange workgroups, Targs... kargs){
    set_argument(0, kargs...);
    c_queue.enqueueNDRangeKernel(def_kernel, offset, threads, workgroups);
    c_queue.finish();
}

size_t sum_weight_elements(const std::vector<std::vector<std::vector<float>>>& weights){
    int sum = 0;
    for(std::vector<std::vector<float>> vv : weights){
        sum += vv.size()*vv[0].size();
    }
    return sum;
}

void create_buffers(const std::vector<std::vector<std::vector<float>>>& weights){
    weight_buffer = cl::Buffer(def_device_context, CL_MEM_READ_ONLY, sizeof(float)*sum_weight_elements(weights));
    input_buffer.push_back(cl::Buffer(def_device_context, CL_MEM_READ_ONLY, sizeof(float)*weights[0].size()));
    output_buffer.push_back(cl::Buffer(def_device_context, CL_MEM_READ_ONLY, sizeof(float)*weights[0].size()))
    for(int i = 0; i < weights.size(); ++i){
        input_buffer.push_back(cl::Buffer(def_device_context, CL_MEM_READ_WRITE, sizeof(float)*weights[i][0].size()));
        output_buffers.push_back(cl::Buffer(def_device_context, CL_MEM_READ_WRITE, sizeof(float)*weights[i][0].size()));
    }
}


