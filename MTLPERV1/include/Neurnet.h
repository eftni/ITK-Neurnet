#ifndef NEURNET_H
#define NEURNET_H
#include "vector"
#include "functional"
#include "random"
#include "iostream"

//std::ostream& operator<<(std::ostream& out){};

class Neurnet
{
    public:
        Neurnet(std::vector<int> layer_count, double learnrate, std::function<double(double)> activator, std::function<double(double)> derivative);
        virtual ~Neurnet();

        std::vector<std::random_device::result_type> get_seed(){return randgen_seeds;}
        std::vector<double> forprop(std::vector<std::vector<uint8_t>> image);
        void backprop(std::vector<double> target, std::vector<double> output);
        std::vector<std::vector<double>> calc_deltas(std::vector<double> target, std::vector<std::vector<double>> outputs)
        //std::ostream& operator<<(std::ostream& out);
    protected:

    private:
        double learning_rate; //!< Coefficient for weight adjustment "eta"
        std::vector<std::vector<std::vector<double>>> weights; //!< Weights between neurons
        std::function<double(double)> act_func; //!< Neuron activation function (usually sigmoid)
        std::function<double(double)> act_func_derivative; //!< Derivative of neuron activation function
        std::vector<std::random_device::result_type> randgen_seeds; //!< Seeds used in mersenne twister for reproducibility
};

#endif // NEURNET_H
