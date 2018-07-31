#include "..//include//KernelFunctor.h"
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
    def_kernel = cl::Kernel(kernel_program, fname.substr(0, fname.length()-3).c_str());
}

KernelFunctor::~KernelFunctor()
{
    //dtor
}






