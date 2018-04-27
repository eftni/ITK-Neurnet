#ifndef LAYER_H
#define LAYER_H
#include "functional"

class Layer
{
    public:
        Layer();
        Layer(int num, std::function<double(double)> act, std::function<double(double)> der, bool d_c);
        virtual ~Layer();

        unsigned int n_number; //!< number of neurons in layer
        std::function<double(double)> activator; //!< Activation function
        std::function<double(double)> derivative; //!< Derivative of activation function
        bool derivative_control;  //!< 0 if derivative is defined with respect to input, 1 if defined with respect to output

    protected:

    private:
};

#endif // LAYER_H
