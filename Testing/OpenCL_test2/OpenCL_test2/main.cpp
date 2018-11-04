#include <iostream>
#include <CL/cl2.hpp>
#include "fstream"
#include "math.h"

class KernelFunctor
{
    public:
        /** Default constructor */
        KernelFunctor();
        KernelFunctor(std::string fname){
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
        KernelFunctor(std::string fname, KernelFunctor k){
            def_platform = k.def_platform;
            def_device = k.def_device;
            def_device_context = k.def_device_context;
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
        /** Default destructor */
        virtual ~KernelFunctor(){}

        template<typename T>
        void set_argument(int num, T arg){
            def_kernel.setArg(num, arg);
        }

        template<typename T, typename... Targs>
        void set_argument(int num, T first, Targs... args){
            def_kernel.setArg(num, first);
            set_argument(num+1, args...);
        }

        template<typename... Targs>
        void operator()(cl::NDRange offset, cl::NDRange threads, cl::NDRange workgroups, Targs... kargs){
            set_argument(0, kargs...);
            c_queue.enqueueNDRangeKernel(def_kernel, offset, threads, workgroups);
            c_queue.finish();
        }

        cl::Context get_context(){return def_device_context;}

        cl::CommandQueue c_queue;

    protected:

    private:
        cl::Platform def_platform;
        cl::Device def_device;
        cl::Context def_device_context;
        cl::Program::Sources kernel_code;
        cl::Program kernel_program;
        cl::Kernel def_kernel;
        std::string fetch_kernel_code(std::string fname){
            std::ifstream fin(fname);
            if(fin.good()){
                return std::string((std::istreambuf_iterator<char>(fin)), std::istreambuf_iterator<char>());
            }else{
                std::cout << fname << ": No such file found!" << std::endl;
                exit(4);
            }
        }
};

void operator/(std::vector<float>& v, float d){
    for(float& f : v){
        f = f/d;
    }
}

void calc_deltas(bool first, int target, std::vector<float> outputs, std::vector<float> w, std::vector<float> deltas_prev, std::vector<float>& deltas_next){
    if(first){
        for(int i = 0; i < deltas_next.size(); ++i){
            if(target == i){
                deltas_next[i] = -(1-outputs[i])*(1-pow(tanh(outputs[i]),2));
            }else{
                deltas_next[i] = -(0-outputs[i])*(1-pow(tanh(outputs[i]),2));
            }
        }
    }else{
        for(int i = 0; i < deltas_next.size(); ++i){
            //std::cout << std::endl << "Delta index: " << i << std::endl << std::endl;
            float acc = 0;
            for(int j = 0; j < deltas_prev.size(); ++j){
                acc += deltas_prev[j]*w[j*deltas_next.size() + i];
                //std::cout << "Deltas_prev: " << deltas_prev[j] << " Weight: " << w[j*deltas_next.size() + i] << std::endl;
            }
            deltas_next[i] = acc*(1-pow(tanh(outputs[i]),2));
            /*std::cout << "Acc: " << acc << " Derivative: " << 1-pow(tanh(outputs[i]),2) << " Output index: " << i
            << " Output: " << outputs[i] << std::endl;
            std::cout << "-----------" << std::endl << std::endl;*/
        }
    }
}

void backprop(std::vector<float> outputs, std::vector<float> deltas, std::vector<float>& new_ws){
    for(int i = 0; i < outputs.size(); ++i){
        for(int j = 0; j < deltas.size(); ++j){
            //new_ws[j*3 + i] = outputs[i] * deltas[j] * 1;
            new_ws[j*5 + i] = outputs[i] * deltas[j] * 1;
            std::cout << "Output: " << outputs[i]  << " Delta: " << deltas[j] << " Result: " << outputs[i] * deltas[j] << std::endl;
            std::cout << "Index: " << j*3 + i << " Correct index? " << j*5 + i << std::endl;
        }
    }
}

int main()
{
    KernelFunctor forprop_kernel("forprop.cl");
    std::vector<float> input = {1,2,3,4,5};
    //input/255;
    std::vector<float> w = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    std::vector<float> output(3, 0);
    std::vector<float> test(5, 0);
    cl::Buffer input_buffer = cl::Buffer(forprop_kernel.get_context(), CL_MEM_READ_WRITE, sizeof(float)*input.size());
    cl::Buffer w_buffer = cl::Buffer(forprop_kernel.get_context(), CL_MEM_READ_WRITE, sizeof(float)*w.size());
    cl::Buffer output_buffer = cl::Buffer(forprop_kernel.get_context(), CL_MEM_READ_WRITE, sizeof(float)*output.size());
    cl::Buffer act_output_buffer = cl::Buffer(forprop_kernel.get_context(), CL_MEM_READ_WRITE, sizeof(float)*output.size());
    forprop_kernel.c_queue.enqueueWriteBuffer(input_buffer, CL_FALSE, 0, sizeof(float)*input.size(), &input[0]);
    forprop_kernel.c_queue.enqueueWriteBuffer(w_buffer, CL_FALSE, 0, sizeof(float)*w.size(), &w[0]);
    forprop_kernel(cl::NullRange, cl::NDRange(3,1), cl::NullRange, input_buffer, 5, w_buffer, 5, output_buffer, 3, 2, act_output_buffer);
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
    KernelFunctor delta_kernel("calc_deltas.cl", forprop_kernel);
    cl::Buffer delta_buffer = cl::Buffer(delta_kernel.get_context(), CL_MEM_READ_WRITE, sizeof(float)*output.size());
    cl::Buffer next_delta_buffer = cl::Buffer(delta_kernel.get_context(), CL_MEM_READ_WRITE, sizeof(float)*input.size());
    cl::Buffer target_buffer = cl::Buffer(delta_kernel.get_context(), CL_MEM_READ_WRITE, sizeof(float)*output.size());
    std::vector<uint8_t> target = {2};
    delta_kernel.c_queue.enqueueWriteBuffer(target_buffer, CL_TRUE, 0, sizeof(float)*target.size(), &target[0]);
    cl::Buffer dummy = cl::Buffer(delta_kernel.get_context(), CL_MEM_READ_WRITE, 0);
    delta_kernel(cl::NullRange, cl::NDRange(3, 1), cl::NullRange, 0, input_buffer, act_output_buffer, w_buffer, 3, target_buffer, 2, dummy, 0, delta_buffer);
    std::vector<float> deltas(3,0);
    delta_kernel.c_queue.enqueueReadBuffer(delta_buffer, CL_TRUE, 0, sizeof(float)*deltas.size(), &deltas[0]);
    std::cout << std::endl << "GPU deltas:" << std::endl;
    for(float f : deltas){
        std::cout << f << std::endl;
    }
    std::cout << std::endl;
    std::vector<float> deltas2(3, 0);
    calc_deltas(1, 2, output, w, {}, deltas2);
    std::cout << "CPU deltas:" << std::endl;
    for(float f : deltas2){
        std::cout << f << std::endl;
    }
    std::cout << std::endl;
    delta_kernel(cl::NullRange, cl::NDRange(5, 1), cl::NullRange, 1, input_buffer, input_buffer, w_buffer, 3, target_buffer, 2, delta_buffer, 3, next_delta_buffer);
    std::vector<float> next_deltas(5,0);
    delta_kernel.c_queue.enqueueReadBuffer(next_delta_buffer, CL_TRUE, 0, sizeof(float)*next_deltas.size(), &next_deltas[0]);
    std::cout << "GPU deltas:" << std::endl;
    for(float f : next_deltas){
        std::cout << f << std::endl;
    }
    std::cout << std::endl;
    std::vector<float> deltas3(5, 0);
    calc_deltas(0, 3, input, w, deltas2, deltas3);
    std::cout << "CPU deltas:" << std::endl;
    for(float f : deltas3){
        std::cout << f << std::endl;
    }

    KernelFunctor backprop_kernel("backprop.cl", forprop_kernel);
    cl::Buffer w_update = cl::Buffer(backprop_kernel.get_context(), CL_MEM_READ_WRITE, sizeof(float)*w.size());
    float learn_rate = 1;
    backprop_kernel(cl::NullRange, cl::NDRange(5,3,1), cl::NullRange, delta_buffer, input_buffer, 5, w_update, 3, learn_rate);
    std::vector<float> new_ws(15, 0);
    backprop_kernel.c_queue.enqueueReadBuffer(w_update, CL_TRUE, 0, sizeof(float)*15, &new_ws[0]);
    std::cout << "GPU backprop:" << std::endl;
    for(float f : new_ws){
        std::cout << f << std::endl;
    }
    backprop(input, deltas2, new_ws);
    std::cout << "CPU backprop:" << std::endl;
     for(float f : new_ws){
        std::cout << f << std::endl;
    }
    return 0;
}
