#ifndef KERNELFUNCTOR_H
#define KERNELFUNCTOR_H
#include <CL/cl2.hpp>

class KernelFunctor
{
    public:
        /** Default constructor */
        KernelFunctor();
        KernelFunctor(std::string fname);
        /** Default destructor */
        virtual ~KernelFunctor();

        template<typename T>
        void set_argument(int num, T arg);

        template<typename T, typename... Targs>
        void set_argument(int num, T first, Targs... Args);

        template<typename... Targs>
        std::vector<float> operator()(Targs... Kargs);

        void create_buffers(const std::vector<std::vector<std::vector<float>>>& weights);

    protected:

    private:
        cl::Platform def_platform;
        cl::Device def_device;
        cl::Context def_device_context;
        cl::Program::Sources kernel_code;
        cl::Program kernel_program;
        cl::CommandQueue c_queue;
        cl::Kernel def_kernel;
        cl::Buffer weight_buffer;
        std::vector<cl::Buffer> input_buffers, output_buffers;
        std::string fetch_kernel_code(std::string fname);
};

#endif // KERNELFUNCTOR_H
