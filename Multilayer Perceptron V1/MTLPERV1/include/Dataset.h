#ifndef DATASET_H
#define DATASET_H
#include "vector"
#include "utility"
#include "stdint.h"
#include "fstream"
#include "stdlib.h"
#include "iostream"

class Dataset
{
    public:
        /** Default constructor */
        Dataset(std::string fname_im, std::string fname_lab);
        /** Default destructor */
        virtual ~Dataset();
        void load_one();
        std::pair<uint32_t, uint32_t> get_size();
        std::vector<std::vector<uint8_t>> get_im(){return curr_im;}
        uint8_t get_label(){return curr_label;}


    protected:

    private:
        uint32_t magic_im; //!< Member variable "magic_im"
        uint32_t magic_lab; //!< Member variable "magic_lab"
        uint32_t num_im; //!< Member variable "num_im"
        uint32_t num_lab; //!< Member variable "num_lab"
        uint32_t sizex; //!< Member variable "sizex"
        uint32_t sizey; //!< Member variable "sizey"
        int index; //!< Member variable "index"
        std::ifstream fin_im; //!< Member variable "fin_im"
        std::ifstream fin_lab; //!< Member variable "fin_lab"
        std::vector<std::vector<uint8_t>> curr_im; //!< Member variable "curr_im"
        uint8_t curr_label; //!< Member variable "curr_label"
};

#endif // DATASET_H
