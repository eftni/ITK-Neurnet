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
        Neurnet(std::vector<Layer> layers, double learnrate, std::vector<std::random_device::result_type> rs = {0,0,0,0,0,0,0,0});
        virtual ~Neurnet();

        std::vector<std::random_device::result_type> get_seed(){return randgen_seeds;}
        std::vector<std::vector<double>> forprop(std::vector<std::vector<uint8_t>> image);
        void backprop(std::vector<double> target, std::vector<std::vector<double>> output,std::vector<std::vector<std::vector<double>>>& weights_update);
        std::vector<std::vector<double>> calc_deltas(std::vector<double> target, std::vector<std::vector<double>> outputs);
        void single_pass(uint8_t label, std::vector<std::vector<uint8_t>> image);
        double train_pass(uint8_t label, std::vector<std::vector<uint8_t>> image, std::vector<std::vector<std::vector<double>>>& weights_update);
        void train_net(Dataset& training, int batchsize);
        void write_to_master();
        void test_net(Dataset& testing);

        //std::ostream& operator<<(std::ostream& out);
    protected:

    private:
        double learning_rate; //!< Coefficient for weight adjustment "eta"
        std::vector<std::random_device::result_type> randgen_seeds; //!< Seeds used in mersenne twister for reproducibility
        std::vector<std::vector<std::vector<double>>> weights; //!< Weights between neurons
        std::vector<std::vector<double>> biases;
        std::vector<Layer> n_layers;
        unsigned int hit; //!< Number of pictures guessed correctly
        unsigned int miss; //!< Number of pictures guessed incorrectly
        std::ofstream logfile; //!< Logfile for training and testing
};

#endif // NEURNET_H
