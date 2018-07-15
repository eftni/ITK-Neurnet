#ifndef KERNELFUNCTOR_H
#define KERNELFUNCTOR_H
#include <CL/cl2.hpp>

class KernelFunctor
{
    public:
        /** Default constructor */
        KernelFunctor();
        KernelFunctor(int files, const char* fnames...);
        /** Default destructor */
        virtual ~KernelFunctor();

        template<typename... Targs>
        void operator()(int kernel_arg_count, Targs... Kargs);

        void create_buffers(const std::vector<std::vector<std::vector>>& weights);

    protected:

    private:
        cl::Platform def_platform;
        cl::Device def_device;
        cl::Context def_device_context;
        cl::Program::Sources kernel_code;
        cl::Program kernel_program;
        cl::CommandQueue c_queue;
        cl::Kernel matrix_math;
        std::string fetch_kernel_code(std::string fname);
};

#endif // KERNELFUNCTOR_H
