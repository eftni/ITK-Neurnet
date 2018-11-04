#ifndef KERNELFUNCTOR_H
#define KERNELFUNCTOR_H
#include <CL/cl2.hpp>

class KernelFunctor
{
    public:
        /** Default constructor */
        KernelFunctor();
        KernelFunctor(std::string fname);
        KernelFunctor(std::string fname, KernelFunctor k);
        /** Default destructor */
        virtual ~KernelFunctor();

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
            c_queue.finish(); //Finishing slows execution down significantly
        }

        cl::Context get_context(){return def_device_context;}

        cl::CommandQueue c_queue;
        size_t max_work_group_size;

    protected:

    private:
        cl::Platform def_platform;
        cl::Device def_device;
        cl::Context def_device_context;
        cl::Program::Sources kernel_code;
        cl::Program kernel_program;
        cl::Kernel def_kernel;
        std::string fetch_kernel_code(std::string fname);
};

#endif // KERNELFUNCTOR_H
