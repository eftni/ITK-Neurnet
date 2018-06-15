#include "Layer.h"
#include "utility"
#include "iostream"
#include "stdlib.h"

Layer::Layer()
{
    //ctor
}

Layer::Layer(int num, act_func_type act) :
n_number(num),
activator(act)
{
    //ctor
}

Layer::~Layer()
{
    //dtor
}

