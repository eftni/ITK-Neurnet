#ifndef NEURNET_H
#define NEURNET_H
#include "vector"
#include "functional"
#include "random"
#include "iostream"
#include "Dataset.h"
#include "Layer.h"
#include "KernelFunctor.h"

class Neurnet
{
    public:
        Neurnet();
        Neurnet(std::vector<Layer> layers, float learnrate, size_t batch, KernelFunctor fp_ker, KernelFunctor delta_ker, KernelFunctor bp_ker, std::vector<std::random_device::result_type> rs = {0,0,0,0,0,0,0,0});
        virtual ~Neurnet();

        /**
        * Returns the seeds used by the random generator for reproducibility
        * @return Array of 8 seeds
        */
        std::vector<std::random_device::result_type> get_seed(){return randgen_seeds;}

        void create_buffers(cl::Context c);

        void GPUtest(std::vector<uint8_t> im, uint8_t lab);

        //std::ostream& operator<<(std::ostream& out);
    protected:

    private:
        float learning_rate; //!< Coefficient for weight adjustment "eta"
        size_t batch_size;
        std::vector<std::random_device::result_type> randgen_seeds; //!< Seeds used in mersenne twister for reproducibility
        //std::vector<std::vector<std::vector<float>>> weights; //!< Weights between neurons
        std::vector<std::vector<float>> weights;
        std::vector<std::vector<float>> biases;
        std::vector<Layer> n_layers;
        std::vector<cl::Buffer> input_buffers, output_buffers, delta_buffers, w_buffers, w_update_buffers;
        KernelFunctor forprop_kernel, delta_kernel, backprop_kernel;
        unsigned int hit; //!< Number of pictures guessed correctly
        unsigned int miss; //!< Number of pictures guessed incorrectly
        std::ofstream logfile; //!< Logfile for training and testing
};

#endif // NEURNET_H
