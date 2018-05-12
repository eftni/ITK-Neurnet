#include "Layer.h"
#include "utility"
#include "iostream"
#include "stdlib.h"

Layer::Layer()
{
    //ctor
}

Layer::Layer(int num, std::function<double(double)> act, std::function<double(double)> der, bool d_c) :
n_number(num),
activator(act),
derivative(der),
derivative_control(d_c)
{
    //ctor
}

Layer::~Layer()
{
    //dtor
}

