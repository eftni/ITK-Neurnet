#ifndef LAYER_H
#define LAYER_H
#include "functional"
#include "math.h"

enum act_func_type {identity = 0, sigmoid = 1, hyp_tan = 2};

class Layer
{
    public:
        Layer();
        Layer(int num, act_func_type act);
        virtual ~Layer();

        unsigned int n_number; //!< number of neurons in layer
        act_func_type activator; //!< Activation function

    protected:

    private:
};

#endif // LAYER_H
