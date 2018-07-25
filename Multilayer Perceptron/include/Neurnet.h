#ifndef NEURNET_H
#define NEURNET_H
#include "vector"
#include "functional"
#include "random"
#include "iostream"
#include "Dataset.h"
#include "Layer.h"

class Neurnet
{
    public:
        Neurnet();
        Neurnet(std::vector<Layer> layers, float learnrate, std::vector<std::random_device::result_type> rs = {0,0,0,0,0,0,0,0});
        virtual ~Neurnet();

        /**
        * Returns the seeds used by the random generator for reproducibility
        * @return Array of 8 seeds
        */
        std::vector<std::random_device::result_type> get_seed(){return randgen_seeds;}

        /**
        * Performs forward propoagation on the currently loaded image.
        * @param image The current image
        * @return The inputs and outputs of every neuron
        */
        std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>> forprop(std::vector<std::vector<uint8_t>> image);

        /**
        * Performs backpropagation training using a set of outputs from a forward pass
        * @param target The expected output on the final layer. Genrated by gen_target() in matrixmath.h
        * @see gen_target()
        * @param output The outputs of every neuron in the network
        * @param weights_update The total sum of weight updates in a batch. (Required due to batch implementation)
        */
        void backprop(std::vector<float> target, const std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>>& ins_outs, std::vector<std::vector<std::vector<float>>>& weights_update, std::vector<std::vector<float>>& bias_update);

        /**
        * Calculates the deltas (neuron-output-independent components of a weight update) for every neuron
        * @param target The expected output on the final layer. Genrated by gen_target() in matrixmath.h
        * @param outputs The outputs of every neuron in the network
        * @return The deltas of every neuron in the network
        */
        std::vector<std::vector<float>> calc_deltas(std::vector<float> target, const std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>>& ins_outs);

        /**
        * Performs a single forward propagation and check if the output is correct
        * @param label The number the current image represents
        * @param image The currently loaded image
        */
        void single_pass(uint8_t label, std::vector<std::vector<uint8_t>> image);

        /**
        * Performs a single forward propagation and trains the network based on the output
        * @param label The number the current image represents
        * @param image The currently loaded image
        * @param weights_update The total sum of weight updates in a batch. (Required due to batch implementation)
        * @return The total error value of the output
        */
        float train_pass(uint8_t label, std::vector<std::vector<uint8_t>> image, std::vector<std::vector<std::vector<float>>>& weights_update, std::vector<std::vector<float>>& bias_update);

        /**
        * Trains the network using one entire dataset
        * @param training The dataset containing the training images and labels
        * @param batchsize The number of images to be processed and averaged before an update
        */
        void train_net(Dataset& training, int batchsize);

        /**
        * Writes the properties of the network to the master file.
        */
        void write_to_master();

        /**
        * Test the network's performance using one entire dataset
        * @param testing The dataset containing the testing images and labels
        */
        void test_net(Dataset& testing);

        //std::ostream& operator<<(std::ostream& out);
    protected:

    private:
        float learning_rate; //!< Coefficient for weight adjustment "eta"
        std::vector<std::random_device::result_type> randgen_seeds; //!< Seeds used in mersenne twister for reproducibility
        std::vector<std::vector<std::vector<float>>> weights; //!< Weights between neurons
        std::vector<std::vector<float>> biases;
        std::vector<Layer> n_layers;
        unsigned int hit; //!< Number of pictures guessed correctly
        unsigned int miss; //!< Number of pictures guessed incorrectly
        std::ofstream logfile; //!< Logfile for training and testing
};

#endif // NEURNET_H
