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
        /** Default constructor */
        Neurnet(int input, std::vector<int> layer_count, int output, double learnrate, std::function<double(double)> activator, std::function<double(double)> derivative);
        /** Default destructor */
        virtual ~Neurnet();
        std::vector<std::random_device::result_type> get_seed(){return randgen_seeds;}
        int forprop(std::vector<std::vector<uint8_t>> image);
        //std::ostream& operator<<(std::ostream& out);
    protected:

    private:
        std::vector<std::vector<double>> layers; //!< Member variable "layers"
        std::vector<std::vector<std::vector<double>>> weights; //!< Member variable "weights"
        double learning_rate;
        std::function<double(double)> act_func; //!< Member variable "act_func"
        std::function<double(double)> act_func_derivative;
        std::vector<std::random_device::result_type> randgen_seeds;
};

#endif // NEURNET_H
